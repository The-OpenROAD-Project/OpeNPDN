import json


class T6_PSI_settings():

    def __init__(self):

        self.work_dir = "./"
        self.temp_json_file = self.work_dir + "input/template_definition.json"
        self.conf_json_file = self.work_dir + "input/tool_config.json"
        self.template_file = self.work_dir + "templates"
        self.cur_map_file = self.work_dir + "output/current_map.csv"
        self.cong_map_file = self.work_dir + "output/congestion_map.csv"
        self.cur_map_process_file = self.work_dir + \
                                    "work/current_map_processed.csv"
        self.map_dir = self.work_dir + "input/current_maps/"
        self.parallel_run_dir = self.work_dir + "work/parallel_runs/"
        self.CNN_data_dir =  self.work_dir + "output/"
        self.checkpoint_dir = self.work_dir +'output/checkpoints/'
        self.checkpoint_file = 'power_grid_ckpt'
        self.normalization_file = 'normalization.json'
        self.checkpoint_dir_wo_cong = self.work_dir +'output/checkpoints_wo_cong/'

        self.template_data = self.load_json(self.temp_json_file)
        self.config = self.load_json(self.conf_json_file)

        self.NUM_VDD = self.config["num_vdd_per_region"]
        self.current_unit = self.config["current_unit"]
        self.num_parallel_runs = self.config["num_parallel_runs"]
        self.num_per_run = self.config["num_per_run"]
        self.num_maps = self.config["num_maps"]
        self.start_maps = self.config["start_maps"]
        self.validation_percent = self.config["validation_percent"]
        self.test_percent = self.config["test_percent"]
        self.current_scale = self.config["current_scaling"]
        self.N_EPOCHS = self.config["N_EPOCHS"]
        self.max_current = self.config["max_current"]
        self.current_map_num_regions = self.config["current_map_num_regions"]

        self.WIDTH_REGION = self.template_data['property']['SIZE_REGION_X']
        self.LENGTH_REGION = self.template_data['property']['SIZE_REGION_Y']
        self.NUM_TEMPLATES = self.template_data['property']['NUM_TEMPLATES']
        self.PDN_layers_ids = self.template_data['property']['PDN_layers']
        self.NUM_LAYERS = self.template_data['property']['NUM_layers']
        self.NUM_REGIONS_X = self.template_data['property']['NUM_REGIONS_X']
        self.NUM_REGIONS_Y = self.template_data['property']['NUM_REGIONS_Y']
        self.NUM_REGIONS = self.NUM_REGIONS_X * self.NUM_REGIONS_Y
        self.LAYERS = self.template_data['layers']
        self.TECH_LAYERS = self.template_data['property']['TECH_layers']

        self.VDD = self.template_data['property']['VDD']
        self.n_c4 = self.NUM_VDD * self.NUM_REGIONS_X * self.NUM_REGIONS_Y
        self.IR_DROP_LIMIT = self.template_data['property']['IR_DROP_LIMIT']

        self.NUM_PDN_LAYERS = len(self.PDN_layers_ids)

    def load_json(self, json_file):
        with open(json_file) as f:
            json_data = json.load(f)
        return json_data
