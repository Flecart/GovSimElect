# shows example on how to change model

MODEL="anthropic/claude-sonnet-4.6" \
GROUP="nl-sonnet46" \
bash scripts/standard/run_iid_stochastic_nl.sh


# shows example on how to change model

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="1-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &


MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="1-nl-prosocial" \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="2-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="2-code-law-prosocial" \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="0-no-contract" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="0-no-contract-prosocial" \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &

wait

MODEL="qwen/qwen3.6-plus" \
GROUP="qwen36plus-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="qwen/qwen3.6-plus" \
GROUP="qwen36plus-nl-prosocial" \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="qwen/qwen3.6-plus" \
GROUP="qwen36plus-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="qwen/qwen3.6-plus" \
GROUP="qwen36plus-code-law-prosocial" \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &

MODEL="qwen/qwen3.6-plus" \
GROUP="qwen36plus-no-contract" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="qwen/qwen3.6-plus" \
GROUP="qwen36plus-no-contract-prosocial" \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &

wait

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-nl-prosocial" \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-code-law-prosocial" \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &
wait

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-no-contract-2" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-no-contract-prosocial-2" \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &
