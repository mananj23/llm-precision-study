# llm-precision-study
#
# The report's table and every figure regenerate from the raw result files with
# one command:  make all
#
# Nothing here touches a GPU. The GPU steps (serving, quantizing, evaluating,
# benchmarking) are deliberately manual, because each one should be started
# inside tmux and watched. See README.md for the order.

PY ?= python3
SYN := results/_synthetic

.PHONY: help
help:
	@echo "Analysis (no GPU needed):"
	@echo "  make all          collect raw -> tables -> figures"
	@echo "  make collect      parse lm-eval + vllm-bench output into tidy files"
	@echo "  make results      build analysis/results.parquet (+ bootstrap intervals)"
	@echo "  make figures      Pareto + long-context plots"
	@echo "  make router       train and evaluate the router against its baselines"
	@echo "  make power        what effect size your benchmarks can resolve"
	@echo ""
	@echo "Testing (no GPU, no real data needed):"
	@echo "  make test         generate synthetic data and run the whole pipeline"
	@echo "  make synthetic    just regenerate the synthetic fixture"
	@echo "  make clean-test   remove the synthetic fixture"
	@echo ""
	@echo "Setup:"
	@echo "  make check        environment / GPU / library check"
	@echo "  make configs      regenerate configs/models/*.yaml"

# ------------------------------------------------------------------ analysis --
.PHONY: all
all: collect results figures
	@echo
	@echo "Done. Table: analysis/results.csv   Figures: report/*.png"

.PHONY: collect
collect:
	$(PY) scripts/collect_logs.py
	-$(PY) scripts/collect_bench.py

.PHONY: results
results:
	$(PY) analysis/build_results.py

.PHONY: figures
figures:
	$(PY) analysis/plot_pareto.py
	-$(PY) analysis/plot_longcontext.py
	-$(PY) analysis/power.py --from-data

.PHONY: power
power:
	$(PY) analysis/power.py                 # worked example, no data needed
	-$(PY) analysis/power.py --from-data    # measured discordance, once you have results

# Router needs the cheap/expensive pair named explicitly -- there is no sensible
# default, and silently picking one would hide a choice that belongs in the report.
CHEAP     ?= llama31-8b__w-fp8__kv-bf16
EXPENSIVE ?= llama31-8b__w-bf16__kv-bf16

.PHONY: router
router:
	$(PY) router/train.py --cheap $(CHEAP) --expensive $(EXPENSIVE) --model both
	@echo
	@echo "Now the honest version, without the task-identity feature:"
	$(PY) router/train.py --cheap $(CHEAP) --expensive $(EXPENSIVE) \
		--model both --no-task-flag --out report/fig_router_no_task_flag.png

# ------------------------------------------------------------------- testing --
.PHONY: synthetic
synthetic:
	$(PY) tests/make_synthetic.py --clean

.PHONY: test
test: synthetic
	@echo "--- bootstrap self-check ---"
	$(PY) analysis/bootstrap.py
	@echo
	@echo "--- collect ---"
	$(PY) scripts/collect_logs.py  --quality-root $(SYN)/quality --out $(SYN)/logs/per_prompt.parquet
	$(PY) scripts/collect_bench.py --system-root  $(SYN)/system  --out $(SYN)/logs/system.parquet
	@echo
	@echo "--- results ---"
	$(PY) analysis/build_results.py --per-prompt $(SYN)/logs/per_prompt.parquet \
		--system $(SYN)/logs/system.parquet --outdir $(SYN)/analysis --price-per-hour 2.49
	@echo
	@echo "--- figures ---"
	$(PY) analysis/plot_pareto.py --results $(SYN)/analysis/results.parquet \
		--out $(SYN)/fig_pareto.png
	$(PY) analysis/plot_longcontext.py --per-prompt $(SYN)/logs/per_prompt.parquet \
		--out $(SYN)/fig_longcontext.png
	@echo
	@echo "--- router ---"
	$(PY) router/train.py --per-prompt $(SYN)/logs/per_prompt.parquet \
		--results $(SYN)/analysis/results.parquet \
		--cheap llama31-8b__w-int4gptq__kv-bf16 \
		--expensive llama31-8b__w-bf16__kv-bf16 \
		--model logistic --out $(SYN)/fig_router.png
	@echo
	@echo "--- leakage guard ---"
	$(PY) tests/test_guards.py
	@echo
	@echo "PIPELINE OK. Every number above came from FAKE data; none of it is a result."

.PHONY: clean-test
clean-test:
	rm -rf $(SYN)

# --------------------------------------------------------------------- setup --
.PHONY: check
check:
	bash scripts/env_check.sh

.PHONY: configs
configs:
	$(PY) scripts/gen_configs.py
