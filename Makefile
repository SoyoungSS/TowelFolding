SHELL := /bin/bash
.DEFAULT_GOAL := help
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SIM_ENV ?= towel-newton
VR_ENV ?= towel-vr
ARGS ?=
VR_PYTHON ?=
HOST_IP ?=
CONDA ?= $(HOME)/miniforge3/bin/conda
ISAAC_ENV ?= dexonomy-isaac
NEWTON_PYTHON ?= $(HOME)/miniforge3/envs/$(SIM_ENV)/bin/python
SIM = conda run --no-capture-output -n "$(SIM_ENV)" python -I -u

.PHONY: help check sim sim-fold sim-vr test test-vr-input certificate verify package bridge-record bridge-replay bridge-check bridge-live
help:
	@echo "make check          Check Newton and CUDA"
	@echo "make sim            Interactive robot, towel and cameras"
	@echo "make sim-fold       Automatic single half-fold"
	@echo "make sim-vr         VR motion input, no headset video"
	@echo "make test           Scene and synthetic VR regressions (GPU required)"
	@echo "make certificate HOST_IP=192.168.1.10   Create local VR TLS files"
	@echo "make test-vr-input  Live local WebXR transport check (port 8012 must be free)"
	@echo "make verify         Verify delivery file hashes and assets"
	@echo "make package        Refresh manifest and create ../newton_towel_sim.tar.gz"
	@echo "make bridge-record  Record a headless Newton fold for Isaac replay"
	@echo "make bridge-replay  Render recorded fold in Isaac (accept EULA beforehand)"
	@echo "make bridge-check   Check trajectory, camera frame mapping and decode all videos"
	@echo "make bridge-live    Start live Newton + Isaac Sim UI (accept EULA beforehand)"
bridge-live:
	@cd "$(ROOT)" && "$(CONDA)" run --no-capture-output -n "$(ISAAC_ENV)" python -I -u scripts/isaac/live_bridge.py --newton-python "$(NEWTON_PYTHON)" $(ARGS)
bridge-record:
	@mkdir -p "$(ROOT)/outputs/bridge"
	@cd "$(ROOT)" && "$(CONDA)" run --no-capture-output -n "$(SIM_ENV)" python -I -u scripts/isaac/phase1_record_newton.py $(ARGS)
bridge-replay:
	@cd "$(ROOT)" && "$(CONDA)" run --no-capture-output -n "$(ISAAC_ENV)" python -I -u scripts/isaac/phase2_replay_isaac.py $(ARGS)
bridge-check:
	@cd "$(ROOT)" && "$(CONDA)" run --no-capture-output -n "$(ISAAC_ENV)" python -I -u scripts/isaac/check_bridge_outputs.py $(ARGS)
check:
	@$(SIM) "$(ROOT)/scripts/check_newton.py" $(ARGS)
sim:
	@cd "$(ROOT)" && $(SIM) "$(ROOT)/simulation/towel_scene.py" $(ARGS)
sim-fold:
	@cd "$(ROOT)" && $(SIM) "$(ROOT)/simulation/towel_scene.py" --auto-fold --num-frames 2400 $(ARGS)
sim-vr:
	@set -e; vr_python='$(VR_PYTHON)'; \
	if [[ -z "$$vr_python" ]]; then vr_python="$$(conda run -n "$(VR_ENV)" python -I -c 'import sys; print(sys.executable)')"; fi; \
	cd "$(ROOT)"; $(SIM) "$(ROOT)/simulation/towel_scene.py" --vr --vr-python "$$vr_python" $(ARGS)
test:
	@$(SIM) "$(ROOT)/scripts/check_newton.py"
	@$(SIM) "$(ROOT)/tests/test_newton_scene.py"
	@$(SIM) "$(ROOT)/tests/test_newton_vr.py"
test-vr-input:
	@conda run --no-capture-output -n "$(VR_ENV)" python -I "$(ROOT)/tests/check_sim_vr_input.py"
certificate:
	@bash "$(ROOT)/scripts/create_certificate.sh" "$(HOST_IP)"
verify:
	@python3 -I "$(ROOT)/scripts/package.py" --verify
package:
	@python3 -I "$(ROOT)/scripts/package.py"

.PHONY: action-replay
action-replay:
	@cd "$(ROOT)" && "$(NEWTON_PYTHON)" scripts/action_replay.py $(ARGS)
