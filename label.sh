# Step 1: 跑 1800 条 train
python -m data.model_agreement_narrativeqa \
  --split train --max_n 1800 \
  --output outputs/label/model_agreement_train_1800.jsonl \
  --base_url "$BASE_URL" --api_key "$API_KEY"
# Step 2: 跑 200 条 validation
python -m data.model_agreement_narrativeqa \
  --split validation --max_n 200 \
  --output outputs/label/model_agreement_validation_200.jsonl \
  --base_url "$BASE_URL" --api_key "$API_KEY"
# Step 3: LLM 语义判断（train）
python -m data.llm_semantic_judge \
  --agreement_file outputs/label/model_agreement_train_1800.jsonl \
  --base_url "$BASE_URL" --api_key "$API_KEY"
# Step 4: LLM 语义判断（validation）
python -m data.llm_semantic_judge \
  --agreement_file outputs/label/model_agreement_validation_200.jsonl \
  --base_url "$BASE_URL" --api_key "$API_KEY"