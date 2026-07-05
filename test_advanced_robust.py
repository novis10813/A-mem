"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amem.memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from amem.reranking import DEFAULT_CROSS_ENCODER_MODEL, build_reranker
from amem import llm_text_parsers as _ltp
from amem.llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
    set_keyword_pruning_mode,
)
from amem.memory_pipeline import (
    MemoryProcessingPipeline,
    PipelineTimingHook,
    merge_timing_summaries,
)
import os
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from amem.load_dataset import load_locomo_dataset
import nltk
from collections import defaultdict
from collections import Counter
import pickle
import random
from tqdm import tqdm
from amem.utils import calculate_metrics, aggregate_metrics
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

logger = logging.getLogger("amem_robust")


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 memory_pipeline=None, reranker=None, rerank_top_n=None,
                 retrieval_mode="embedding"):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            pipeline=memory_pipeline,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
            retrieval_mode=retrieval_mode,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.last_retrieval_info = {}

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10, rerank_query=None):
        return self.memory_system.find_related_memories_raw(
            content, k=k, rerank_query=rerank_query
        )

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(
            keywords, k=self.retrieve_k, rerank_query=question
        )
        self.last_retrieval_info = dict(self.memory_system.last_retrieval_info)
        self.last_retrieval_info["query_keywords"] = keywords
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def merge_sample_outputs(sample_outputs):
    """Merge per-sample worker outputs in deterministic sample order."""
    results = []
    all_metrics = []
    all_categories = []
    category_counts = Counter()
    error_num = 0
    sample_timing = []

    for sample_output in sorted(sample_outputs, key=lambda item: item["sample_idx"]):
        results.extend(sample_output["results"])
        all_metrics.extend(sample_output["metrics"])
        all_categories.extend(sample_output["categories"])
        category_counts.update(sample_output["category_counts"])
        error_num += sample_output.get("error_num", 0)
        sample_timing.append({
            "sample_idx": sample_output["sample_idx"],
            "stage_timing": sample_output.get("stage_timing", {}),
        })

    return {
        "results": results,
        "metrics": all_metrics,
        "categories": all_categories,
        "category_counts": category_counts,
        "total_questions": len(all_metrics),
        "error_num": error_num,
        "stage_timing": merge_timing_summaries(
            sample["stage_timing"] for sample in sample_timing
        ),
        "sample_timing": sample_timing,
    }


def evaluate_sample(sample_idx: int, sample, model: str, backend: str,
                    retrieve_k: int, temperature_c5: float,
                    sglang_host: str, sglang_port: int,
                    memories_dir: str, allow_categories,
                    eval_logger: logging.Logger,
                    show_progress: bool = True,
                    rerank_mode: str = "off",
                    rerank_model: str = DEFAULT_CROSS_ENCODER_MODEL,
                    rerank_top_n: Optional[int] = None,
                    rerank_batch_size: int = 32):
    """Evaluate one LoCoMo sample with an isolated agent and cache files."""
    timing_hook = PipelineTimingHook()
    memory_pipeline = MemoryProcessingPipeline(hooks=[timing_hook])
    reranker = build_reranker(rerank_mode, rerank_model, rerank_batch_size)
    agent = RobustAdvancedMemAgent(
        model, backend, retrieve_k, temperature_c5,
        sglang_host, sglang_port, memory_pipeline, reranker, rerank_top_n,
    )

    memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
    retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
    retriever_cache_embeddings_file = os.path.join(
        memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    )

    if os.path.exists(memory_cache_file):
        eval_logger.info(f"Loading cached memories for sample {sample_idx}")
        with open(memory_cache_file, 'rb') as f:
            cached_memories = pickle.load(f)
        agent.memory_system.memories = cached_memories
        if os.path.exists(retriever_cache_file):
            eval_logger.info(f"Found retriever cache files for sample {sample_idx}")
            agent.memory_system.retriever = agent.memory_system.retriever.load(
                retriever_cache_file, retriever_cache_embeddings_file
            )
        else:
            eval_logger.info(f"No retriever cache found for sample {sample_idx}, loading from memory")
            agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                cached_memories, 'all-MiniLM-L6-v2'
            )
        eval_logger.info(f"Successfully loaded {len(cached_memories)} memories for sample {sample_idx}")
    else:
        eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")

        all_turns = [
            (turns.date_time, turn)
            for _, turns in sample.conversation.sessions.items()
            for turn in turns.turns
        ]
        with tqdm(all_turns, desc=f"[Sample {sample_idx}] Building memories",
                  unit="turn", dynamic_ncols=True, disable=not show_progress) as pbar:
            for turn_datatime, turn in pbar:
                conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                agent.add_memory(conversation_tmp, time=turn_datatime)
                mem_count = len(agent.memory_system.memories)
                pbar.set_postfix(memories=mem_count)

        memories_to_cache = agent.memory_system.memories
        with open(memory_cache_file, 'wb') as f:
            pickle.dump(memories_to_cache, f)
        agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
        eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories for sample {sample_idx}")

    eval_logger.info(f"Processing sample {sample_idx}")

    results = []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)
    total_questions = 0

    eligible_qas = [qa for qa in sample.qa if int(qa.category) in allow_categories]
    with tqdm(eligible_qas, desc=f"[Sample {sample_idx}] Answering QAs",
              unit="qa", dynamic_ncols=True, disable=not show_progress) as pbar:
        for qa in pbar:
            total_questions += 1
            category_counts[qa.category] += 1
            pbar.set_postfix(cat=qa.category, total=total_questions)

            prediction, user_prompt, raw_context = agent.answer_question(
                qa.question, qa.category, qa.final_answer
            )

            prediction = parse_plain_text_answer(prediction)

            eval_logger.info(f"Sample {sample_idx} question {total_questions}: {qa.question}")
            eval_logger.info(f"Prediction: {prediction}")
            eval_logger.info(f"Reference: {qa.final_answer}")
            eval_logger.info(f"User Prompt: {user_prompt}")
            eval_logger.info(f"Category: {qa.category}")
            eval_logger.info(f"Raw Context: {raw_context}")

            metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
            }

            all_metrics.append(metrics)
            all_categories.append(qa.category)

            results.append({
                "sample_id": sample_idx,
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.final_answer,
                "category": qa.category,
                "metrics": metrics,
                "retrieval_info": agent.last_retrieval_info,
            })

            if total_questions % 10 == 0:
                eval_logger.info(f"Processed {total_questions} questions for sample {sample_idx}")

    return {
        "sample_idx": sample_idx,
        "results": results,
        "metrics": all_metrics,
        "categories": all_categories,
        "category_counts": dict(category_counts),
        "error_num": 0,
        "stage_timing": timing_hook.summary(),
    }


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     keyword_pruning_mode: str = "nltk", max_workers: int = 10,
                     rerank_mode: str = "off",
                     rerank_model: str = DEFAULT_CROSS_ENCODER_MODEL,
                     rerank_top_n: Optional[int] = None,
                     rerank_batch_size: int = 32):
    """Evaluate the robust agent on the LoComo dataset."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if rerank_mode != "off" and (rerank_top_n is None or rerank_top_n < retrieve_k):
        raise ValueError("rerank_top_n must be >= retrieve_k when reranking is enabled")

    # Configure keyword pruning mode before any memory operations
    set_keyword_pruning_mode(keyword_pruning_mode)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_kp{keyword_pruning_mode}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")
    eval_logger.info(f"Keyword pruning mode: {keyword_pruning_mode}")
    eval_logger.info(f"Rerank mode: {rerank_mode}")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}_{}".format(backend, model, keyword_pruning_mode),
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]

    worker_count = max(1, min(max_workers, len(samples)))
    eval_logger.info(f"Sample worker count: {worker_count}")

    if worker_count == 1:
        sample_outputs = [
            evaluate_sample(
                sample_idx, sample, model, backend, retrieve_k, temperature_c5,
                sglang_host, sglang_port, memories_dir, allow_categories,
                eval_logger, show_progress=True,
                rerank_mode=rerank_mode,
                rerank_model=rerank_model,
                rerank_top_n=rerank_top_n,
                rerank_batch_size=rerank_batch_size,
            )
            for sample_idx, sample in enumerate(samples)
        ]
    else:
        sample_outputs = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    evaluate_sample,
                    sample_idx,
                    sample,
                    model,
                    backend,
                    retrieve_k,
                    temperature_c5,
                    sglang_host,
                    sglang_port,
                    memories_dir,
                    allow_categories,
                    eval_logger,
                    False,
                    rerank_mode,
                    rerank_model,
                    rerank_top_n,
                    rerank_batch_size,
                )
                for sample_idx, sample in enumerate(samples)
            ]
            with tqdm(as_completed(futures), total=len(futures),
                      desc="Samples", unit="sample", dynamic_ncols=True) as pbar:
                for future in pbar:
                    sample_outputs.append(future.result())

    merged = merge_sample_outputs(sample_outputs)
    results = merged["results"]
    all_metrics = merged["metrics"]
    all_categories = merged["categories"]
    category_counts = merged["category_counts"]
    total_questions = merged["total_questions"]
    error_num = merged["error_num"]
    stage_timing = merged["stage_timing"]
    sample_timing = merged["sample_timing"]

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "keyword_pruning_mode": keyword_pruning_mode,
        "rerank_mode": rerank_mode,
        "rerank_model": rerank_model if rerank_mode != "off" else None,
        "rerank_top_n": rerank_top_n if rerank_mode != "off" else None,
        "rerank_batch_size": rerank_batch_size if rerank_mode != "off" else None,
        "sample_workers": worker_count,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "stage_timing": stage_timing,
        "sample_timing": sample_timing,
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    eval_logger.info("Aggregate Stage Timing:")
    for stage_name, stats in stage_timing.items():
        eval_logger.info(
            "  %s: count=%d total=%.4fs avg=%.4fs min=%.4fs max=%.4fs",
            stage_name,
            stats["count"],
            stats["total_seconds"],
            stats["avg_seconds"],
            stats["min_seconds"],
            stats["max_seconds"],
        )

    eval_logger.info("Per-Sample Stage Timing:")
    for sample in sample_timing:
        eval_logger.info("  Sample %s:", sample["sample_idx"])
        for stage_name, stats in sample["stage_timing"].items():
            eval_logger.info(
                "    %s: count=%d total=%.4fs avg=%.4fs",
                stage_name,
                stats["count"],
                stats["total_seconds"],
                stats["avg_seconds"],
            )

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--keyword_pruning_mode", type=str, default="nltk",
                        choices=["none", "simple", "nltk"],
                        help="Keyword pruning mode: 'none' (raw LLM keywords, no filtering), "
                             "'simple' (exact-match grounding, no NLTK stemming), "
                             "or 'nltk' (PorterStemmer derivational-variant matching)")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum number of LoCoMo samples to evaluate in parallel")
    parser.add_argument("--rerank-mode", choices=["off", "cross_encoder"], default="off",
                        help="Optional second-stage reranker for robust retrieval")
    parser.add_argument("--rerank-model", default=DEFAULT_CROSS_ENCODER_MODEL,
                        help="CrossEncoder model for --rerank-mode cross_encoder")
    parser.add_argument("--rerank-top-n", type=int, default=50,
                        help="Similarity candidate count before reranking")
    parser.add_argument("--rerank-batch-size", type=int, default=32,
                        help="CrossEncoder prediction batch size")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if args.rerank_batch_size < 1:
        raise ValueError("rerank_batch_size must be at least 1")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port,
        args.keyword_pruning_mode, args.max_workers,
        args.rerank_mode, args.rerank_model, args.rerank_top_n,
        args.rerank_batch_size,
    )


if __name__ == "__main__":
    main()
