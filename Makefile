# EnhanceShortcut - task automation (gcp-cloudrun-microservices skill pattern)

.PHONY: help local deploy

help:
	@echo "EnhanceShortcut"
	@echo "  make local    - run services locally (voice_enhance on :8081, CPU)"
	@echo "  make deploy   - build + deploy all services to Cloud Run (GPU for voice_enhance)"

local:
	./deploy_local.sh

deploy:
	./deploy_cloud_run.sh
