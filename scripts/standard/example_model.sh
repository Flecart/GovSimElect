# shows example on how to change model

MODEL="anthropic/claude-sonnet-4.6" \
GROUP="nl-sonnet46" \
bash scripts/standard/run_iid_stochastic_nl.sh


# shows example on how to change model
MODEL="openai/gpt-5.4-mini" \
NAME="gpt-5.4-mini" \
GROUP="1-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="gpt-5.4-mini" \
GROUP="1-nl-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="gpt-5.4-mini" \
GROUP="2-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="gpt-5.4-mini" \
GROUP="2-code-law-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="gpt-5.4-mini" \
GROUP="0-no-contract" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="openai/gpt-5.4-mini" \
NAME="gpt-5.4-mini" \
GROUP="0-no-contract-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &

wait

MODEL="qwen/qwen3.6-plus" \
NAME="qwen3.6-plus" \
GROUP="1-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="qwen3.6-plus" \
GROUP="1-nl-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="qwen3.6-plus" \
GROUP="2-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="qwen3.6-plus" \
GROUP="2-code-law-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="qwen3.6-plus" \
GROUP="0-no-contract" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="qwen/qwen3.6-plus" \
NAME="qwen3.6-plus" \
GROUP="0-no-contract-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &

wait

MODEL="mistralai/mistral-small-2603" \
NAME="mistral-small-2603" \
GROUP="1-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="mistral-small-2603" \
GROUP="1-nl-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="mistral-small-2603" \
GROUP="2-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="mistral-small-2603" \
GROUP="2-code-law-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="mistral-small-2603" \
GROUP="0-no-contract" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="mistralai/mistral-small-2603" \
NAME="mistral-small-2603" \
GROUP="0-no-contract-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &

wait

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="1-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &


MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="1-nl-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="2-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="2-code-law-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="0-no-contract" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="0-no-contract-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &


wait

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-nl" \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-nl-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_nl_prosocial.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-code-law" \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-code-law-prosocial" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_code_law_prosocial.sh &
wait

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-no-contract-2" \
bash scripts/standard/run_iid_stochastic_no_contract.sh &

MODEL="mistralai/mistral-small-3.2-24b-instruct" \
GROUP="mistral-small-3.2-24b-no-contract-prosocial-2" \
PROSOCIAL_COUNT=5 \
bash scripts/standard/run_iid_stochastic_no_contract_prosocial.sh &





# gpt-4o sweep: prosocial count from 0 to 5 for NL, code-law, and no-contract
for prosocial in 0 1 2 3 4 5; do
  MODEL="openai/gpt-4o" \
  NAME="gpt-4o" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_nl.sh &

  MODEL="openai/gpt-4o" \
  NAME="gpt-4o" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_code_law.sh &

  MODEL="openai/gpt-4o" \
  NAME="gpt-4o" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_no_contract.sh &
done

wait


MODEL="openai/gpt-4o" \
NAME="gpt-4o" \
GROUP="1-nl-p1" \
PROSOCIAL_COUNT=1 \
bash scripts/standard/run_iid_stochastic_nl.sh &

MODEL="openai/gpt-4o" \
NAME="gpt-4o" \
GROUP="2-code-law-p1" \
PROSOCIAL_COUNT=1 \
bash scripts/standard/run_iid_stochastic_code_law.sh &

MODEL="openai/gpt-4o" \
NAME="gpt-4o" \
GROUP="0-no-contract-p1" \
PROSOCIAL_COUNT=1 \
bash scripts/standard/run_iid_stochastic_no_contract.sh &



for prosocial in 0 1 2 3 4 5; do
  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_nl.sh &

  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_code_law.sh &

  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_no_contract.sh &
done

wait

for prosocial in 0 1 2 3 4 5; do
  MODEL="openai/gpt-5.4" \
  NAME="gpt-5.4" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_nl.sh &
  MODEL="openai/gpt-5.4" \
  NAME="gpt-5.4" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_code_law.sh &
  MODEL="openai/gpt-5.4" \
  NAME="gpt-5.4" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_no_contract.sh &
done
wait



for prosocial in 0 1 2 3 4 5; do
  MODEL="openai/gpt-5.4" \
  NAME="sto/gpt-5.4" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &
  MODEL="openai/gpt-5.4" \
  NAME="sto/gpt-5.4" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &
  MODEL="openai/gpt-5.4" \
  NAME="sto/gpt-5.4" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done
wait



for prosocial in 0 1 2 3 4 5; do
  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &

  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &

  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done

wait



MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="test" \
PROSOCIAL_COUNT=0 \
bash scripts/standard/run_iid_stochastic_nl.sh


  MODEL="openai/gpt-5.4-mini" \
  NAME="gpt-5.4-mini" \
  GROUP="test-p0" \
  PROSOCIAL_COUNT=0 \
  bash scripts/standard/run_iid_stochastic_nl.sh &


    MODEL="openai/gpt-5.4-mini" \
  NAME="sto/gpt-5.4-mini" \
  GROUP="1-nl-p0" \
  PROSOCIAL_COUNT=0 \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &


for prosocial in 0 1 2 3 4 5; do
  MODEL="google/gemma-4-31b-it" \
  NAME="gemma-4-31b" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_nl.sh &

  MODEL="google/gemma-4-31b-it" \
  NAME="gemma-4-31b" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_code_law.sh &

  MODEL="google/gemma-4-31b-it" \
  NAME="gemma-4-31b" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/standard/run_iid_stochastic_no_contract.sh &
done

for prosocial in 0 1 2 3 4 5; do
  MODEL="google/gemma-4-31b-it" \
  NAME="sto/gemma-4-31b" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &

  MODEL="google/gemma-4-31b-it" \
  NAME="sto/gemma-4-31b" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &

  MODEL="google/gemma-4-31b-it" \
  NAME="sto/gemma-4-31b" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done


for prosocial in 0 1 2 3 4 5; do
  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &

  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &

  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done

for prosocial in 0 1 2 3 4 5; do
  MODEL="openai/gpt-5.4" \
  NAME="sto/gpt-5.4" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &
  MODEL="openai/gpt-5.4" \
  NAME="sto/gpt-5.4" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &
  MODEL="openai/gpt-5.4" \
  NAME="sto/gpt-5.4" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done


for prosocial in 0 1 2 3 4 5; do
  MODEL="openai/gpt-4o" \
  NAME="sto/gpt-4o" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &
  MODEL="openai/gpt-4o" \
  NAME="sto/gpt-4o" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &
  MODEL="openai/gpt-4o" \
  NAME="sto/gpt-4o" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done

for prosocial in 0 1 2 3 4 5; do
  MODEL="openai/gpt-5.4-mini" \
  NAME="sto/gpt-5.4-mini" \
  GROUP="1-nl-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh &
  MODEL="openai/gpt-5.4-mini" \
  NAME="sto/gpt-5.4-mini" \
  GROUP="2-code-law-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh &
  MODEL="openai/gpt-5.4-mini" \
  NAME="sto/gpt-5.4-mini" \
  GROUP="0-no-contract-p${prosocial}" \
  PROSOCIAL_COUNT="$prosocial" \
  bash scripts/stochastic/run_iid_stochastic_no_contract.sh &
done



  MODEL="openai/gpt-5.4-mini" \
  NAME="sto/gpt-5.4-mini" \
  GROUP="2-code-law-p2" \
  PROSOCIAL_COUNT="2" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh

 
  MODEL="openai/gpt-4o" \
  NAME="sto/gpt-4o" \
  GROUP="1-nl-ptest" \
  PROSOCIAL_COUNT="" \
  bash scripts/stochastic/run_iid_stochastic_nl.sh


  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="2-code-law-p0" \
  PROSOCIAL_COUNT="0" \
  bash scripts/stochastic/run_iid_stochastic_code_law.sh
MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP="0-no-contract-p0" \
PROSOCIAL_COUNT=0 \
bash scripts/stochastic/run_iid_stochastic_no_contract.sh &



  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="2-code-law-p0" \
  PROSOCIAL_COUNT="0" \
  bash scripts/standard/run_iid_stochastic_code_law.sh


  MODEL="x-ai/grok-4.1-fast" \
  NAME="grok-4.1-fast" \
  GROUP="1-nl-ptest" \
  PROSOCIAL_COUNT="0" \
  bash scripts/standard/run_iid_stochastic_nl.sh