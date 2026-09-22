.PHONY: clean test build healthcheck preview

clean:
	docker run --rm -v $$(pwd)/outputs:/out busybox sh -c "rm -rf /out/TN /out/MU /out/QA /out/.boundaries 2>&1; mkdir -p /out && touch /out/.gitkeep; ls -la /out"
	rm -rf /tmp/opencode
	mkdir -p /tmp/opencode
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "cleaned outputs/caches/pyc"

test:
	python -m py_compile scripts/*.py scripts/local_engine/*.py s2sr/*.py tests/*.py
	python tests/test_pipeline_units.py

build:
	docker compose build

healthcheck:
	docker compose run --rm s2sr-cpu python -c "from pathlib import Path; assert Path('models/s2sr-v3.0.0.pt').exists(); print('healthcheck OK')"

preview:  # deflated preview for sharing (keeps archive untouched)
	@echo "usage: make preview MS=outputs/TN/.../MS.tif"
	@test -n "$(MS)" || (echo "set MS=path/to/MS.tif" && exit 1)
	gdal_translate -co COMPRESS=DEFLATE -co PREDICTOR=2 -co TILED=YES "$(MS)" "$$(dirname $(MS))/MS_preview.tif"
	@echo "preview at $$(dirname $(MS))/MS_preview.tif"
