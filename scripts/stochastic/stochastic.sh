
MODEL="openai/gpt-5.4-mini" \
GROUP="gpt54-mini-nl" \
bash scripts/stochastic/run_iid_stochastic_nl.sh &


MODEL="openai/gpt-5.4-mini" \
GROUP="gpt54-mini-nl-prosocial" \
bash scripts/stochastic/run_iid_stochastic_nl_prosocial.sh &

MODEL="openai/gpt-5.4-mini" \
GROUP="gpt54-mini-code-law" \
bash scripts/stochastic/run_iid_stochastic_code_law.sh &

MODEL="openai/gpt-5.4-mini" \
GROUP="gpt54-mini-code-law-prosocial" \
bash scripts/stochastic/run_iid_stochastic_code_law_prosocial.sh &

MODEL="openai/gpt-5.4-mini" \
GROUP="gpt54-mini-no-contract" \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &

MODEL="openai/gpt-5.4-mini" \
GROUP="gpt54-mini-no-contract-prosocial" \
bash scripts/stochastic/run_iid_stochastic_no_contract_prosocial.sh &

wait


MODEL="x-ai/grok-4.1-fast" \
GROUP="sto-grok41-nl" \
bash scripts/stochastic/run_iid_stochastic_nl.sh &


MODEL="x-ai/grok-4.1-fast" \
GROUP="sto-grok41-nl-prosocial" \
bash scripts/stochastic/run_iid_stochastic_nl_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
GROUP="sto-grok41-code-law" \
bash scripts/stochastic/run_iid_stochastic_code_law.sh &

MODEL="x-ai/grok-4.1-fast" \
GROUP="sto-grok41-code-law-prosocial" \
bash scripts/stochastic/run_iid_stochastic_code_law_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
GROUP="sto-grok41-no-contract" \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &

MODEL="x-ai/grok-4.1-fast" \
GROUP="sto-grok41-no-contract-prosocial" \
bash scripts/stochastic/run_iid_stochastic_no_contract_prosocial.sh &

wait