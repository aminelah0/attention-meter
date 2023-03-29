#################### ENV & PACKAGE ACTIONS ###################
install_env:
	@pyenv virtualenv 3.10.6 attention-env
	@pyenv local attention-env
	@pip install --upgrade pip
	@pip install ipython
	@pip install ipykernel

install_package:
	@pip install -e .

reinstall_package:
	@pip uninstall -y attention || :
	@pip install -e .


################### DATA SOURCES ACTIONS ################
create_data_folder:
	@mkdir attention_data/
	@mkdir attention_data/00_inputs/
	@mkdir attention_data/99_outputs/
	@mkdir attention_data/00_inputs/01_frames/
	@mkdir attention_data/00_inputs/02_video/
	@mkdir attention_data/00_inputs/99_known_faces/
	@mkdir attention_data/99_outputs/00_summary/
	@mkdir attention_data/99_outputs/01_detection/
	@mkdir attention_data/99_outputs/02_face_crops/
	@mkdir attention_data/99_outputs/03_face_mesh/
	@mkdir attention_data/99_outputs/04_attention/
	@mkdir attention_data/99_outputs/05_recognition/

delete_inputs:
	@rm -rf attention_data/00_inputs/01_frames/*
	@rm -rf attention_data/00_inputs/02_video/*

delete_outputs:
	@rm -rf attention_data/99_outputs/00_summary/*
	@rm -rf attention_data/99_outputs/01_detection/*
	@rm -rf attention_data/99_outputs/02_face_crops/*
	@rm -rf attention_data/99_outputs/03_face_mesh/*
	@rm -rf attention_data/99_outputs/04_attention/*
	@rm -rf attention_data/99_outputs/05_recognition/*
	@rm -rf attention_data/99_outputs/attention_output.csv


################### PROJECT INTERFACE ################
PERIOD_SEC = 1

video2frames:
	@python -c "from attention.interface.main_local import video2frames; video2frames(period_sec=$(PERIOD_SEC))"

rename_frames:
	@python -c "from attention.interface.main_local import rename_frames; rename_frames(period_sec=$(PERIOD_SEC))"

train_recognition:
	@python -c "from attention.interface.main_local import train_recognition; train_recognition()"

generate_outputs:
	@python -c "from attention.interface.main_local import generate_outputs; generate_outputs()"
