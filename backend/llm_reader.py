import logging
import torch
from typing import Any

from reader.candidates import AnswerCandidate

logger = logging.getLogger(__name__)

class LocalLLMReader:
    def __init__(self, model_id: str = "LiquidAI/LFM2.5-2.6B") -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            logger.error("transformers and bitsandbytes are required for LocalLLMReader")
            raise

        logger.info(f"Loading local LLM {model_id} in 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="cuda:0",
        )
        logger.info("Local LLM loaded successfully!")

    def predict_direct(self, question: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý AI bằng Tiếng Việt. Hãy trả lời các câu hỏi một cách ngắn gọn, chính xác và trực tiếp nhất có thể. Không giải thích dài dòng."},
            {"role": "user", "content": question}
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Hệ thống: {messages[0]['content']}\nNgười dùng: {messages[1]['content']}\nTrợ lý:"
        
        return self._generate(prompt, method="llm_direct")

    def predict_rag(self, question: str, passages: list[str]) -> dict[str, Any]:
        # Nối các passage lại với nhau để LLM có đủ thông tin (thay vì chỉ lấy đoạn đầu tiên)
        context = "\n---\n".join(passages[:5]) if passages else ""
        if len(context) > 2000:
            context = context[:2000] + "..."
            
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý AI bằng Tiếng Việt. Hãy đọc thông tin cung cấp và trả lời câu hỏi một cách ngắn gọn, chính xác. Không giải thích thêm."},
            {"role": "user", "content": f"Thông tin:\n{context}\n\nCâu hỏi: {question}"}
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Hệ thống: {messages[0]['content']}\nNgười dùng: {messages[1]['content']}\nTrợ lý:"
            
        return self._generate(prompt, method="llm_rag")

    def _generate(self, prompt: str, method: str) -> dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024, # Tăng max_new_tokens để model trả lời đầy đủ
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Loại bỏ phần tư duy (reasoning) của model nếu có thẻ </think>
        if '</think>' in answer:
            answer = answer.split('</think>')[-1].strip()
        
        return {
            "text": answer,
            "method": method,
            "score": 1.0,
            "start": -1,
            "end": -1,
        }
