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

reset_frames:
	@rm -rf attention_data/frames/*

reset_output:
	@rm -rf attention_data/output_bbox/*
	@rm -rf attention_data/face_crops/*
	@rm -rf attention_data/output_mesh/*
	@rm -rf attention_data/output_attention/*
	@rm -rf attention_data/output_recognition/*
	@rm -rf attention_data/output_summary/*

reset_frames_test:
	@rm -rf attention_data_test/frames/*

reset_output_test:
	@rm -rf attention_data_test/output_bbox/*
	@rm -rf attention_data_test/face_crops/*
	@rm -rf attention_data_test/output_mesh/*
	@rm -rf attention_data_test/output_attention/*
	@rm -rf attention_data_test/output_recognition/*
	@rm -rf attention_data_test/output_summary/*
