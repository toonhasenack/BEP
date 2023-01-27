import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sociophysicsDataHandler import SociophysicsDataHandler
import os

def filter_on_length(df, min_length = 10):
    traj_length = df.groupby(by = ["tracked_object"])["tracked_object"].count()
    df_length = df.join(traj_length, on='tracked_object', rsuffix='_trajectory_length')
    return df_length[df_length["tracked_object_trajectory_length"] > min_length].reset_index(drop = True)

def vel(df):
    vel = df.groupby(by = ["tracked_object"])
    df_vel = df.join(vel[["x_pos", "y_pos"]].diff()/10, on='tracked_object', rsuffix='_dot')
    df_vel["speed"] = (df_vel["x_pos_dot"]**2 + df_vel["y_pos_dot"]**2)**(1/2)
    return df_vel

def transition(df, v_tresh = 0.1, dx = 10):
    filt_data = filter_on_length(df.copy())
    filt_data["x_pos"] = filt_data["x_pos"] /1000 //dx
    filt_data["y_pos"] = filt_data["y_pos"] /1000 //dx
    
    loc_low = np.where(filt_data["speed"] < v_tresh)[0]
    loc_high = np.where(filt_data["speed"] >= v_tresh)[0]
    static = filt_data.loc[loc_low].groupby(by = ["date_time_utc", "x_pos", "y_pos"])["tracked_object"].nunique()
    static = static.rename("<Nstat>")
    dynamic = filt_data.loc[loc_high].groupby(by = ["date_time_utc", "x_pos", "y_pos"])["tracked_object"].nunique()
    dynamic = dynamic.rename("<Ndyn>")
    correlation = pd.concat([static, dynamic], axis = 1)
    correlation = correlation.dropna()
    return correlation

def save(n,N_avg,N_std,bins):
    N_avg_save = N_avg.copy()/n.copy()
    N_avg_save[np.isnan(N_avg)] = 0
    N_std_save = N_std.copy()/n.copy()
    N_std_save[np.isnan(N_std)] = 0

    np.savetxt("Result/N_avg.csv", N_avg_save, delimiter = ",")
    np.savetxt("Result/N_std.csv", N_std_save, delimiter = ",")
    np.savetxt("Result/n.csv", n, delimiter = ",")
def run():
    dh = SociophysicsDataHandler()
    os.mkdir("Result")
    string = 'ehv/platform2.1/'
    files = dh.list_files(string)["name"]
    N_tresh = 10
    min_file = 306
    bins = 50

    N_avg = np.zeros([bins])
    N_std = np.zeros([bins])
    n = np.zeros([bins])
    for f in files[min_file:]:
        subfiles = dh.list_files(string + f)["name"]
        for g in subfiles:
            dh.fetch_prorail_data_from_path(string + f + '/' + g, verbose = False)
            if dh.df["tracked_object"].nunique() > N_tresh:
                df = vel(dh.df)
                correlation = transition(df)
                Ns = correlation.groupby(by="<Nstat>")["<Ndyn>"]
                N_avg[Ns.mean().index.astype("int")] += Ns.mean()
                N_std[Ns.std().index.astype("int")] += Ns.std()/len(Ns)
                n[Ns.mean().index.astype("int")] += 1
        save(n, N_avg, N_std, bins)

if __name__ == "__main__":
    run()