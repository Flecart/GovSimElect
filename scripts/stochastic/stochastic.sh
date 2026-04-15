
# shows example on how to change model
MODEL="openai/gpt-5.4-mini" \
NAME="sto-gpt-5.4-mini" \
GROUP="1-nl" \
bash scripts/stochastic/run_iid_stochastic_nl.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="sto-gpt-5.4-mini" \
GROUP="1-nl-prosocial" \
bash scripts/stochastic/run_iid_stochastic_nl_prosocial.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="sto-gpt-5.4-mini" \
GROUP="2-code-law" \
bash scripts/stochastic/run_iid_stochastic_code_law.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="sto-gpt-5.4-mini" \
GROUP="2-code-law-prosocial" \
bash scripts/stochastic/run_iid_stochastic_code_law_prosocial.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="sto-gpt-5.4-mini" \
GROUP="0-no-contract" \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="sto-gpt-5.4-mini" \
GROUP="0-no-contract-prosocial" \
bash scripts/stochastic/run_iid_stochastic_no_contract_prosocial.sh &



MODEL="qwen/qwen3.6-plus" \
NAME="sto-qwen3.6-plus" \
GROUP="1-nl" \
bash scripts/stochastic/run_iid_stochastic_nl.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="sto-qwen3.6-plus" \
GROUP="1-nl-prosocial" \
bash scripts/stochastic/run_iid_stochastic_nl_prosocial.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="sto-qwen3.6-plus" \
GROUP="2-code-law" \
bash scripts/stochastic/run_iid_stochastic_code_law.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="sto-qwen3.6-plus" \
GROUP="2-code-law-prosocial" \
bash scripts/stochastic/run_iid_stochastic_code_law_prosocial.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="sto-qwen3.6-plus" \
GROUP="0-no-contract" \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="sto-qwen3.6-plus" \
GROUP="0-no-contract-prosocial" \
bash scripts/stochastic/run_iid_stochastic_no_contract_prosocial.sh &



MODEL="mistralai/mistral-small-2603" \
NAME="sto-mistral-small-2603" \
GROUP="1-nl" \
bash scripts/stochastic/run_iid_stochastic_nl.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="sto-mistral-small-2603" \
GROUP="1-nl-prosocial" \
bash scripts/stochastic/run_iid_stochastic_nl_prosocial.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="sto-mistral-small-2603" \
GROUP="2-code-law" \
bash scripts/stochastic/run_iid_stochastic_code_law.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="sto-mistral-small-2603" \
GROUP="2-code-law-prosocial" \
bash scripts/stochastic/run_iid_stochastic_code_law_prosocial.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="sto-mistral-small-2603" \
GROUP="0-no-contract" \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="sto-mistral-small-2603" \
GROUP="0-no-contract-prosocial" \
bash scripts/stochastic/run_iid_stochastic_no_contract_prosocial.sh &



MODEL="x-ai/grok-4.1-fast" \
NAME="sto-grok-4.1-fast" \
GROUP="1-nl" \
bash scripts/stochastic/run_iid_stochastic_nl.sh &


MODEL="x-ai/grok-4.1-fast" \
NAME="sto-grok-4.1-fast" \
GROUP="1-nl-prosocial" \
bash scripts/stochastic/run_iid_stochastic_nl_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="sto-grok-4.1-fast" \
GROUP="2-code-law" \
bash scripts/stochastic/run_iid_stochastic_code_law.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="sto-grok-4.1-fast" \
GROUP="2-code-law-prosocial" \
bash scripts/stochastic/run_iid_stochastic_code_law_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="sto-grok-4.1-fast" \
GROUP="0-no-contract" \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="sto-grok-4.1-fast" \
GROUP="0-no-contract-prosocial" \
bash scripts/stochastic/run_iid_stochastic_no_contract_prosocial.sh &

wait
