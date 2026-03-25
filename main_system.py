# /home/ljc/ntire/main_system.py


# /home/ljc/ntire/main_system.py
import os
import torch
import subprocess
import shutil


class NTIREFinalSystem:
    def __init__(self):
        self.unified_pth = "/home/ljc/ntire/weights/final_unified_weights.pth"
        self.temp_dir = "/home/ljc/ntire/weights/temp_weights"
        self.final_output_root = "/home/ljc/ntire/results"
        self.difix_model_pth = "/home/ljc/ntire/checkpoints/difix_hf"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.all_data = torch.load(self.unified_pth, map_location='cpu')

    def _execute(self, cmd, cwd, env_gpu):
        print(f"\n执行指令: {cmd}")
        # 加上 conda run -n ipc 确保去雾环境正确
        wrapped_cmd = f"conda run -n ipc {cmd}"
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(env_gpu)
        subprocess.run(wrapped_cmd, shell=True, cwd=cwd, env=my_env)

    def run_all(self, model_path):
        model_path = os.path.abspath(model_path)
        scene_prefix = os.path.basename(model_path)
        hazy_dir = os.path.join(model_path, "test", "hazy")

        # --- 路径规划 ---
        scene_res_dir = os.path.join(self.final_output_root, scene_prefix)  # 场景根目录
        pre_difix = os.path.join(scene_res_dir, "pre_difix")  # 中转站1
        dehazed_temp = os.path.join(scene_res_dir, "dehazed_temp")  # 中转站2 (存放未滤波的去雾图)
        final_submission = os.path.join(scene_res_dir, "final_submission")  # 终点站

        os.makedirs(pre_difix, exist_ok=True)
        os.makedirs(dehazed_temp, exist_ok=True)
        os.makedirs(final_submission, exist_ok=True)

        # --- 1. UDP (koharu, natsume) ---
        if scene_prefix in ["koharu", "natsume"]:
            udp_w = os.path.join(self.temp_dir, "udp_tmp.ckpt")
            torch.save(self.all_data['weights_udp'], udp_w)
            udp_cmd = f"python test.py --model_name {scene_prefix} --data_dir {model_path} --test_model {udp_w} --save_image True"
            self._execute(udp_cmd, "/home/ljc/ntire/models_repo/UDPNet/Dehazing/ITS", 2)
            # 搬运到 pre_difix
            udp_out = f"/home/ljc/ntire/models_repo/UDPNet/Dehazing/ITS/results/{scene_prefix}/test"
            for f in os.listdir(udp_out):
                if f.endswith(".png"): shutil.copy(os.path.join(udp_out, f), os.path.join(pre_difix, f))

        # --- 2. OneRestore (hinoki) ---
        elif scene_prefix == "hinoki":
            one_w = os.path.join(self.temp_dir, "one_tmp.tar")
            torch.save(self.all_data['weights_onerestore'], one_w)
            one_raw_out = os.path.join(scene_res_dir, "one_raw_tmp")
            one_cmd = f"python test.py --embedder-model-path /home/ljc/ntire/models_repo/onerestore/ckpts/embedder_model.tar --restore-model-path {one_w} --input {hazy_dir} --output {one_raw_out}"
            self._execute(one_cmd, "/home/ljc/ntire/models_repo/onerestore", 2)
            # 搬运到 pre_difix
            for f in os.listdir(one_raw_out):
                if f.endswith(".png"): shutil.copy(os.path.join(one_raw_out, f), os.path.join(pre_difix, f))

        # --- 3. Difix 扩散增强 (针对 K, N, H) ---
        if scene_prefix in ["koharu", "natsume", "hinoki"]:
            difix_out = os.path.join(scene_res_dir, "difix_res_tmp")
            difix_cmd = f"python run.py --model-dir {self.difix_model_pth} --input-dir {pre_difix} --output-dir {difix_out} --steps 1 --timestep 199 --max-side 0"
            self._execute(difix_cmd, "/home/ljc/ntire/models_repo", 2)
            # 【关键修改】：结果汇总到 dehazed_temp
            for f in os.listdir(difix_out):
                shutil.copy(os.path.join(difix_out, f), os.path.join(dehazed_temp, f))

        # --- 4. IPC (midori) ---
        elif scene_prefix == "midori":
            ipc_w = os.path.join(self.temp_dir, "ipc_tmp.pth")
            torch.save(self.all_data['weights_ipc'], ipc_w)
            ipc_out = os.path.join(scene_res_dir, "ipc_tmp")
            ipc_cmd = f"python inference.py -i {hazy_dir} -o {ipc_out} --predictor_path {ipc_w} --critic_path /home/ljc/ntire/weights/critic.pth"
            self._execute(ipc_cmd, "/home/ljc/ntire/models_repo/IPC_Dehaze", 2)
            # 【关键修改】：结果汇总到 dehazed_temp
            for f in os.listdir(ipc_out):
                shutil.copy(os.path.join(ipc_out, f), os.path.join(dehazed_temp, f))

        # --- 5.  (futaba, shirohana, tsubaki) ---
        elif scene_prefix in ["futaba", "shirohana", "tsubaki"]:
            # 【关键修改】：Enhancer 直接输出到 dehazed_temp
            refine_cmd = f"python Enhancer.py --weights {self.unified_pth} --input {hazy_dir} --output {dehazed_temp}"
            self._execute(refine_cmd, "/home/ljc/ntire/models_repo", 2)

        # --- 6. 最终滤波后处理 ---
        # 输入: dehazed_temp (集齐了所有去雾图), 输出: final_submission
        print(f"=== 运行 滤波后处理 ===")
        lvbo_cmd = f"python lvbo.py --input {dehazed_temp} --output {final_submission}"
        self._execute(lvbo_cmd, "/home/ljc/ntire/models_repo", 2)

        # --- 7. 彻底清理中间文件 (可选) ---
        # 如果你希望场景目录下“只有”final_submission，就取消下面几行的注释
        shutil.rmtree(pre_difix)
        shutil.rmtree(dehazed_temp)
        if os.path.exists(os.path.join(scene_res_dir, "one_raw_tmp")): shutil.rmtree(os.path.join(scene_res_dir, "one_raw_tmp"))
        if os.path.exists(os.path.join(scene_res_dir, "difix_res_tmp")): shutil.rmtree(os.path.join(scene_res_dir, "difix_res_tmp"))
        if os.path.exists(os.path.join(scene_res_dir, "ipc_tmp")): shutil.rmtree(os.path.join(scene_res_dir, "ipc_tmp"))

        print(f"✨ [场景 {scene_prefix}] 处理完成！结果仅保存在: {final_submission}")


