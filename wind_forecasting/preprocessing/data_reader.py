import os
import re
import datatable as dt
import yaml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import cycle

class DataReader:
    # def __init__():
    #     pass

    def get_results_data(self, results_dirs):
        # from whoc.wind_field.WindField import first_ord_filter
        results_dfs = {}
        for results_dir in results_dirs:
            case_family = os.path.split(os.path.basename(results_dir))[-1]
            for f, fn in enumerate([fn for fn in os.listdir(results_dir) if ".csv" in fn]):
                seed = int(re.findall(r"(?<=seed\_)(\d*)", fn)[0])
                case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0]

                df = dt.fread(os.path.join(results_dir, fn))

                case_tag = f"{case_family}_{case_name}"
                if case_tag not in results_dfs:
                    results_dfs[case_tag] = [df]
                else:
                    results_dfs[case_tag].append(df)

            results_dfs[case_tag] = dt.rbind(results_dfs[case_tag])
        self.results_dfs = results_dfs
        return results_dfs
    
    def plot_yaw_power_ts(self, data_df, save_path, include_yaw=True, include_power=True, controller_dt=None):
        data_df = data_df.to_pandas()
        colors = sns.color_palette(palette='Paired')

        turbine_wind_direction_cols = sorted([col for col in data_df.columns if "TurbineWindDir_" in col])
        turbine_power_cols = sorted([col for col in data_df.columns if "TurbinePower_" in col])
        yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" in col])

        for seed in sorted(np.unique(data_df["WindSeed"])):
            fig, ax = plt.subplots(int(include_yaw + include_power), 1, sharex=True, figsize=(15.12, 7.98))
            ax = np.atleast_1d(ax)

            seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
            
            if include_yaw:
                ax_idx = 0
                ax[ax_idx].plot(seed_df["Time"], seed_df["FreestreamWindDir"], label="Freestream wind dir.", color="black")
                ax[ax_idx].plot(seed_df["Time"], seed_df["FilteredFreestreamWindDir"], label="Filtered freestream wind dir.", color="black", linestyle="--")
                
            # Direction
            for t, (wind_dir_col, power_col, yaw_col, color) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols, cycle(colors))):
                
                if include_yaw:
                    ax_idx = 0
                    ax[ax_idx].plot(seed_df["Time"], seed_df[yaw_col], color=color, label="T{0:01d} yaw setpoint".format(t), linestyle=":")
                    
                    if controller_dt is not None:
                        [ax[ax_idx].axvline(x=_x, linestyle=(0, (1, 10)), linewidth=0.5) for _x in np.arange(0, seed_df["Time"].iloc[-1], controller_dt)]

                if include_power:
                    next_ax_idx = (1 if include_yaw else 0)
                    if t == 0:
                        ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[power_col] / 1e3, color=color, label="T{0:01d} power".format(t))
                    else:
                        ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e3, 
                                        seed_df[turbine_power_cols[:t]].sum(axis=1)  / 1e3,
                            color=color, label="T{0:01d} power".format(t))
            
            if include_power:
                next_ax_idx = (1 if include_yaw else 0)
                ax[next_ax_idx].plot(seed_df["Time"], seed_df[turbine_power_cols].sum(axis=1) / 1e3, color="black", label="Farm power")
        
            if include_yaw:
                ax_idx = 0
                ax[ax_idx].set(title="Wind Direction / Yaw Angle [$^\\circ$]", xlim=(0, int((seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]) // 1)), ylim=(245, 295))
                ax[ax_idx].legend(ncols=2, loc="lower right")
                if not include_power:
                    ax[ax_idx].set(xlabel="Time [s]", title="Turbine Powers [MW]")
            
            if include_power:
                next_ax_idx = (1 if include_yaw else 0)
                ax[next_ax_idx].set(xlabel="Time [s]", title="Turbine Powers [MW]", ylim=(0, None))
                ax[next_ax_idx].legend(ncols=2, loc="lower right")

            results_dir = os.path.dirname(save_path)
            fig.suptitle("_".join([os.path.basename(results_dir), data_df["CaseName"].iloc[0].replace('/', '_'), "yaw_power_ts", f"seed{seed}"]))
            fig.savefig(save_path.replace(".png", f"seed{seed}.png"))
        # fig.show()
        return fig, ax

    def plot_data(self, results_dirs):
        
        for r, results_dir in enumerate(results_dirs):
            input_filenames = [fn for fn in os.listdir(results_dir) if "input_config" in fn]
            # input_case_names = [re.findall(r"(?<=case_)(.*)(?=.yaml)", input_fn)[0] for input_fn in input_filenames]
            # data_filenames = sorted([fn for fn in os.listdir(results_dir) if ("time_series_results" in fn 
            #                         and re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] in input_case_names)], 
            #                         key=lambda data_fn: input_case_names.index(re.findall(r"(?<=case_)(.*)(?=_seed)", data_fn)[0]))
            
            # for f, data_fn in enumerate(data_filenames):
            for f, input_fn in enumerate(input_filenames):
                case_family = os.path.basename(results_dir)
                # data_case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", data_fn)[0]
                # input_fn = f"input_config_case_{data_case_name}.yaml"
                case_name = re.findall(r"(?<=input_config_case_)(.*)(?=.yaml)", input_fn)[0]
                
                # with open(os.path.join(results_dir, input_fn), 'r') as fp:
                #     input_config = yaml.safe_load(fp)

                full_case_name = f"{case_family}_{case_name}"
                df = self.results_dfs[full_case_name]
                fig, _ = self.plot_yaw_power_ts(data_df=df, save_path=os.path.join(results_dir, f"yaw_power_ts_{full_case_name}.png"))
    
    
if __name__ == "__main__":
    data_dir = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies/lut"

    data_reader = DataReader()
    data_reader.get_results_data([data_dir])
    data_reader.plot_data([data_dir])
