SHELL = bash
PC=python3
DEF_FILE= /home/vchhabria/alpha-release/flow/results/nangate45/aes_cipher_top_bkp/aes.def
POW_FILE= /home/vchhabria/alpha-release/flow/results/nangate45/aes_cipher_top_bkp/aes.pwr.rpt
LEF_FILE=  /home/vchhabria/alpha-release/flow/results/nangate45/aes_cipher_top_bkp/aes.lef
CONGEST_RPT= /home/vchhabria/alpha-release/flow/results/nangate45/aes_cipher_top_bkp/aes.congest.rpt

TERM_SHELL= $(shell echo "$$0")
COMMAND =  $(shell source install.sh)
.PHONY: work templates 
.SILENT: build all

CONGESTION_ENABLED = 0

ifeq (${CONGESTION_ENABLED}, 0)
	CONGESTION_COMMAND = "no_congestion"
	CONGEST_RPT = "congestion_report_invalid"
else
	CONGESTION_COMMAND = ""
endif

clean: work
	rm -rf ./work

work:  
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs 

maps:
	mkdir -p input/current_maps &&\
	$(PC) ./scripts/create_training_set.py 

templates: 
	$(PC) ./src/create_template.py &&\
	$(PC) ./src/eliminate_templates.py

data:
	$(PC) ./scripts/run_batch_iterative.py ${CONGESTION_COMMAND}

training: 
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND}

parse_inputs:
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGEST_RPT} ${CONGESTION_COMMAND}

predict:
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND}

release:
	$(PC) ./scripts/clean_release.py

install:
	$(PC) -m venv PDN-opt
ifeq ($(TERM_SHELL), bash)
	( \
		source PDN-opt/bin/activate; \
		pip3 install -r requirements.txt --no-cache-dir; \
		bash -c "source PDN-opt/bin/activate"; \
	)
else
	( \
		source PDN-opt/bin/activate.csh; \
		pip3 install -r requirements.txt --no-cache-dir; \
		tcsh -c "source PDN-opt/bin/activate.csh"; \
	)
endif

build:
	echo "****************************************************************" &&\
	echo "****** Running scripts within PDN-opt virtual environment ******" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "***** Extracting training sets and existing CNN checkpoints ****" &&\
	echo "****************************************************************" &&\
	$(PC) ./scripts/build.py &&\
	echo "****************************************************************" &&\
	echo "************************ Build completed ***********************" &&\
	echo "****************************************************************"

all:
	rm -rf ./work &&\
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs &&\
	mkdir -p input/current_maps &&\
	echo "****************************************************************" &&\
	echo "************* Creating the defined templates *******************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/create_template.py &&\
	$(PC) ./src/eliminate_templates.py && \
	echo "****************************************************************" &&\
	echo "************* Creating the maps for SA *************************" &&\
	echo "****************************************************************" &&\
	$(PC) ./scripts/create_training_set.py &&\
	echo "****************************************************************" &&\
	echo "*** Running simulated annealing for training data collection ***" &&\
	echo "****************************************************************" &&\
	$(PC) ./scripts/run_batch_iterative.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "***************** Simulated annealing completed ****************" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "********* Beginning CNN training with the golden data **********" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "************* Creating the testcase current map ****************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "***************** Using CNN to synthesize PDN ******************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "*** CNN-based PDN synthesized and stored in template_map.txt ***" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "************* Running IR drop solver on the PDN ****************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/IR_map_generator.py

train:
	rm -rf ./work &&\
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs &&\
	mkdir -p input/current_maps &&\
	$(PC) ./scripts/create_training_set.py &&\
	$(PC) ./src/create_template.py &&\
	$(PC) ./src/eliminate_templates.py &&\
	$(PC) ./scripts/run_batch_iterative.py ${CONGESTION_COMMAND} &&\
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND}

inference:
	mkdir -p work &&\
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGEST_RPT} ${CONGESTION_COMMAND} && \
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND} && \
	$(PC) ./src/IR_map_generator.py

get_ir:
	mkdir -p work &&\
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGEST_RPT} ${CONGESTION_COMMAND} &&\
	$(PC) ./src/IR_map_generator.py &&\
	$(PC) ./src/voltage_gen.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ./output/IR_drop.csv

test:
	pytest
