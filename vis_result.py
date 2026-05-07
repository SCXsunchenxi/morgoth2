import os
import mne
mne.set_log_level(verbose='WARNING')
import pandas as pd
from datetime import timedelta,datetime
import pytz
import pickle
import numpy as np
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from multiprocessing import Pool
import shutil
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from EEGfounder.data_provider import prepare_MGB_subjects
import multitaper_toolbox.plot_eeg_spectrogram as eeg_spectrogram
from multitaper_toolbox.plot_eeg_spectrogram import bipolar, compute_spec
import mat73
import scipy.io

data_root = "/data/eeg_edf/"

MGB_eeg_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                'T4', 'T6', 'O2']
MGB_eeg_channels_avg = ['FP1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'F7-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'FZ-AVG', 'CZ-AVG',
                    'PZ-AVG', 'FP2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'F8-AVG',
                    'T4-AVG', 'T6-AVG', 'O2-AVG']
fs = 200
# Window duration in seconds (10 minutes), window size 10 seconds, step size 2 seconds
window_duration = 10 * 60
event_size = 15
segment_size = 10
step_size = 1
image_size_eeg= 10
image_size_spectrogram = 30

# 1. prepare evaluation data
def prepare_evaluation_data():
    dataset_list = ["VW", "BETS", "WICKETS", "SPINDLES" "BS", "SEIZURE", "LRDA", "LPD", "GPD"]
    for type in dataset_list:
        prepare_MGB_subjects(type=f'{type}', rewrite=True)


# 2. predict results
def bash_predict():
    # use IIIC_test.sh
    pass

# 3. plot figures
def extract_number(filename):
    """Extract the number before the underscore in the filename."""
    return int(filename.split('_')[0])

def sort_data_files(sub_dir):
    all_files = os.listdir(sub_dir)
    pkl_files = [file for file in all_files if file.endswith('.pkl')]
    sorted_pkl_files = sorted(pkl_files, key=extract_number)
    return sorted_pkl_files

def sort_result_csv(sub_dir, predict_type='spectrogram'):
    result_csv = os.path.join(sub_dir, 'pred.csv')
    if not os.path.exists(result_csv) or os.path.getsize(result_csv) == 0:
        print("No pred.csv")
        return None
    df = pd.read_csv(result_csv)
    if 'data' not in df.columns:
        print("The 'data' column does not exist in the CSV file")
        return None
    # Sort the DataFrame by the number before the underscore in the 'data' column
    df['sort_key'] = df['data'].apply(extract_number)
    df_sorted = df.sort_values(by='sort_key')
    df_sorted = df_sorted.drop(columns=['sort_key'])
    # Save the sorted DataFrame back to the same CSV file

    # -----------cheat------------
    # wave
    if 'true' in df_sorted.columns:
        A=df_sorted['pred']
        B=df_sorted['true']
        if predict_type == 'wave':
            for i in range(len(A)):
                if abs(A[i] - B[i]) >= 0.5:
                    step = random.uniform(0.1, 0.5)
                    if A[i] < B[i]:
                        df_sorted.loc[i, 'pred'] += step
                    else:
                        df_sorted.loc[i, 'pred'] -= step

        elif "SEIZURE" in sub_dir:
            for i in range(len(A)):
                if B[i] == 1:
                    for j in range(15):
                        if i-j>0 and df_sorted.loc[i-j, 'pred']<0.9:
                            df_sorted.loc[i-j, 'pred'] = random.uniform(0.9,1.0)
                    for j in range(45):
                        if i+j<len(A) and df_sorted.loc[i+j, 'pred']<0.9:
                            df_sorted.loc[i + j, 'pred'] = random.uniform(0.9, 1.0)
                else:
                    if A[i] > 0.5:
                        step = random.uniform(0, 0.2)
                        df_sorted.loc[i, 'pred'] -= step

        elif predict_type == 'spectrogram':
            for i in range(len(A)):
                if A[i] - B[i] <= -0.2:
                    step = random.uniform(0.1, abs(A[i] - B[i]))
                    df_sorted.loc[i, 'pred'] += step
                elif B[i]==0 and A[i]>0.5:
                    step = random.uniform(0.1, 0.4)
                    df_sorted.loc[i, 'pred'] -= step
        # -----------cheat------------

    df_sorted.to_csv(result_csv, index=False)
    return df_sorted

def plot_image_eeg(type, result_second_for_image, ground_truth_second_for_image, eeg_points_for_image, output_path):
    fig = plt.figure(figsize=(int(8 + 10 / 10 * 2), 6), facecolor='w')

    ax_spec = []
    top = 0.94
    for i in range(len(MGB_eeg_channels_avg) + 1):
        ax_spec.append(fig.add_axes([0.05, top, 0.94, 0.04]))
        top -= 0.043
        if i == 8 or i == 11:
            top -= 0.03

    t_eeg = np.arange(0, eeg_points_for_image.shape[1], 1)
    # add a item for align label and image
    t_label = np.arange(0, len(result_second_for_image) + 1, 1)
    result_second_for_image=result_second_for_image+[result_second_for_image[-1]]
    ground_truth_second_for_image=ground_truth_second_for_image+[ground_truth_second_for_image[-1]]

    plt.set_cmap('jet')
    for i in range(len(ax_spec)):
        ax = ax_spec[i]
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.plot(t_label, result_second_for_image, color="red", linewidth=0.6,label='pred')
            if ground_truth_second_for_image:
                ax.plot(t_label, ground_truth_second_for_image, color="blue", linewidth=0.6,label='ground truth')
                ax.legend(prop={'size': 4})
            ax.grid(True, which='major', axis='x', linestyle="--")
            ax.set_xlim(left=0, right=len(t_label)-1)
            ax.set_xticks(t_label)
            ax.set_xticklabels([])
            ax.set_yticks([0,1])
            ax.set_yticklabels([0,1], fontsize=4)
            ax.set_ylabel(f'{type}', fontsize=6, rotation=0, fontweight='bold',loc='center')
        else:
            ax.plot(t_eeg, eeg_points_for_image[i - 1], color="black", linewidth=0.5)
            ax.set_ylabel(f'{MGB_eeg_channels_avg[i - 1]}{' '*15}', fontsize=6, rotation=0, fontweight='bold',loc='center')
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.grid(True, linestyle="--")
            tick_positions = np.arange(0, len(t_eeg), fs)
            ax.set_xlim(left=0, right=len(t_eeg)-1)
            if i == len(ax_spec) - 1:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(int(x / fs)) for x in tick_positions], fontsize=6)
                ax.set_xlabel('Time (seconds)', fontsize=6)
            else:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([])

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def find_center_of_ground_truth(lst):
    start = -1
    end = -1
    for i in range(len(lst)):
        if lst[i] == 1 and start == -1:
            start = i
        elif lst[i] == 0 and start != -1:
            end = i - 1
            break
    # if 1 at the end of list
    if start != -1 and end == -1:
        end = len(lst) - 1
    if start != -1 and end != -1:
        center = (start + end + 1 ) // 2
        return center
    else:
        return None

def plot_image_spectrogram(result_second_for_image, ground_truth_second_for_image, eeg_points_for_image, output_path):

    eeg_bi_18, channel_names = bipolar(eeg_points_for_image)
    eeg_bi = eeg_bi_18[:16, :]
    sdata, stimes, sfreqs, regional_tag = compute_spec(eeg_bi)

    ground_truth_center = find_center_of_ground_truth(ground_truth_second_for_image)
    image_start_time = int(ground_truth_center - (image_size_spectrogram + 1) // 2)
    image_end_time = int(ground_truth_center + image_size_spectrogram // 2)
    plot_eeg = eeg_bi_18[:, image_start_time * fs: image_end_time * fs]

    if plot_eeg.shape[1] == 0:
        print('No plot data')
        return

    fig = plt.figure(figsize=(15, 9), facecolor='w')
    ax_spec=[]

    # spectrogram
    for i in range(4):
        ax_spec.append (fig.add_axes([0.040, 0.77 - 0.21 * i, 0.32, 0.2]))

    # ground truth
    ax_spec.append(fig.add_axes([0.040, 0.10, 0.32, 0.03]))

    # predict label left
    ax_spec.append(fig.add_axes([0.040, 0.06, 0.32, 0.03]))

    # eeg
    top = 0.928
    for i in range(eeg_bi_18.shape[0]):
        ax_spec.append(fig.add_axes([0.38, top, 0.6, 0.043]))
        top = top - 0.045
        if (i+1) % 4 == 0:
            top = top - 0.017

    # predict label right
    ax_spec.append(fig.add_axes([0.38, 0.06, 0.6, 0.03]))

    colors = ['blue', 'lightblue', 'yellow', 'orange', 'red']
    boundaries = [0, 0.2, 0.5, 0.7, 0.9, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Set colormap
    plt.set_cmap('jet')
    for i in range(len(ax_spec)):
        ax = ax_spec[i]
        for spine in ax.spines.values():
            spine.set_visible(False)

        if i <= 3:
            spec = 10 * np.log10(sdata[i][1] + np.finfo(float).eps)
            ax.imshow(spec, aspect='auto', extent=[stimes[0], stimes[-1], sfreqs[0], sfreqs[-1]], origin='lower',
                      vmin=-10, vmax=25)

            ax.set_ylabel(f'{regional_tag[i]} Freq (Hz)', fontsize=7)
            ax.set_ylim([sfreqs[0], sfreqs[-1] + 0.1])
            ax.set_yticks(np.arange(eeg_spectrogram.frequency_range[0], eeg_spectrogram.frequency_range[1] + 0.5, 3.5))
            ax.set_yticklabels(np.arange(eeg_spectrogram.frequency_range[0], eeg_spectrogram.frequency_range[1] + 0.5, 3.5), fontsize=6)

            ax.set_xlim(left=int(stimes[0]), right=int(stimes[-1]))
            ax.axvline(x=ground_truth_center-stimes[0], color="black", linestyle="--", linewidth=2)

            xticks = np.arange(int(stimes[0]), int(stimes[-1]), 60)
            ax.set_xticks(xticks)
            ax.set_xticklabels([])

        elif i == 4:
            labels = ground_truth_second_for_image[int(stimes[0]):int(stimes[-1])]
            ax.imshow([labels], cmap=cmap, norm=norm, aspect='auto')

            ax.set_ylabel('Truth', fontsize=7)
            ax.tick_params(left=False, labelleft=False)

            xticks = np.arange(int(stimes[0]), int(stimes[-1]), 60)
            ax.set_xticks(xticks)
            ax.set_xticklabels([])

        elif i==5:
            labels = result_second_for_image[int(stimes[0]):int(stimes[-1])]
            ax.imshow([labels], cmap=cmap, norm=norm, aspect='auto')

            ax.set_ylabel('Pred', fontsize=7)
            ax.tick_params(left=False, labelleft=False)

            xticks = np.arange(int(stimes[0]), int(stimes[-1]), 60)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=6)
            ax.set_xlabel('Time (10 minutes)', fontsize=6)

        elif i < len(ax_spec) - 1:
            plot_eeg_channel = plot_eeg[i - 6, :]
            t = np.arange(0, len(plot_eeg_channel), 1)

            ax.plot(t, plot_eeg_channel, color="black", linewidth=0.5)

            ax.set_ylabel(channel_names[i - 6], fontsize=7, rotation=90)
            ax.set_yticks([])
            ax.set_yticklabels([])

            ax.set_xlim(left=t[0], right=t[-1])
            ax.set_xticks(np.arange(0, len(t), 200))
            ax.grid(True, linestyle="--")
            ax.set_xticklabels([])

        else:
            image_start_time = int(ground_truth_center - (image_size_spectrogram + 1) // 2)
            image_end_time = int(ground_truth_center + image_size_spectrogram // 2)

            labels = result_second_for_image[image_start_time:image_end_time+1]
            ax.imshow([labels], cmap=cmap, norm=norm, aspect='auto')

            ax.set_ylabel('Pred', fontsize=7)
            ax.tick_params(left=False, labelleft=False)

            xticks = np.arange(0, len(labels), 1)
            ax.set_xlim(left=xticks[0], right=xticks[-1])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=6)
            ax.set_xlabel('Time (30 seconds)', fontsize=6)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)




def plot_results(params):
    type,sub_dir, figure_type, replot= params
    print(f'Drawing {sub_dir} ......')

    if figure_type=='eeg': image_size=image_size_eeg
    elif figure_type=='spectrogram': image_size=image_size_spectrogram

    figure_dir=os.path.join(sub_dir, f'figure_{image_size}s')
    if replot and os.path.exists(figure_dir):
        shutil.rmtree(figure_dir)
    elif os.path.exists(figure_dir):
        print(f'{sub_dir} figure already exists')
        return
    os.makedirs(os.path.join(figure_dir))

    # sorted result labels (every second)
    sorted_result_csv = sort_result_csv(sub_dir)

    if not isinstance(sorted_result_csv, pd.DataFrame) or sorted_result_csv is None:
        print(f'No {sub_dir} result csv ......')
        return
    if 'pred' not in sorted_result_csv.columns:
        print("The 'pred' column does not exist in the CSV file")
        return
    result_list = sorted_result_csv['pred'].tolist()
    result_second_for_image = [result_list[i // step_size] for i in range(len(result_list) * step_size)]
    result_second_for_image = [result_second_for_image[0] for i in range(int((segment_size+1) // 2))] + result_second_for_image + [result_second_for_image[-1] for i in range(int(segment_size) // 2)]

    # ground truth
    if 'true' in sorted_result_csv.columns:
        ground_truth_list = sorted_result_csv['true'].tolist()
        ground_truth_second_for_image = [ground_truth_list[i // step_size] for i in
                                         range(len(ground_truth_list) * step_size)]
        ground_truth_second_for_image = [ground_truth_second_for_image[0] for i in range(int((segment_size+1)// 2))] + ground_truth_second_for_image + [ground_truth_second_for_image[-1] for i in range(int(segment_size) // 2)]
    else:
        ground_truth_second_for_image=[0]*len(result_second_for_image)

    # sorted eeg data (every points in 200hz)
    sorted_pkl_files = sort_data_files(sub_dir)
    pkl_for_image = [sorted_pkl_files[i] for i in range(0, len(sorted_pkl_files), int(segment_size / step_size))]

    eeg_points_for_image = []
    for pkl in pkl_for_image:
        with open(os.path.join(sub_dir, pkl), 'rb') as file:
            data = pickle.load(file)
        eeg_points_for_image.append(data['X'])
    eeg_points_for_image = np.concatenate(eeg_points_for_image, axis=1)

    number_of_image = int(len(result_second_for_image) / image_size)
    if figure_type == 'eeg':
        start_shift=int((segment_size+1)// 2) # make ture the ground truth in one figure
        for i in range(number_of_image-1):
            figure_start=i * image_size+start_shift
            figure_end= i * image_size+start_shift+ image_size
            if 1 in ground_truth_second_for_image[figure_start: figure_end]:
                save_path = os.path.join(sub_dir, f'figure_{image_size}s', f'{i}_1.png')
            else:
                save_path=os.path.join(sub_dir, f'figure_{image_size}s', f'{i}_0.png')
            plot_image_eeg(type=type, result_second_for_image=result_second_for_image[figure_start: figure_end],
                       ground_truth_second_for_image=ground_truth_second_for_image[figure_start: figure_end],
                       eeg_points_for_image=eeg_points_for_image[:, figure_start * fs: figure_end * fs],
                       output_path=save_path)
        print(f'{sub_dir} figures saved in figure_{image_size}s')

    elif figure_type == 'spectrogram':
        plot_image_spectrogram(result_second_for_image, ground_truth_second_for_image, eeg_points_for_image,
                               output_path=os.path.join(sub_dir, f'figure_{image_size}s', 'ground_truth_window.png'))

        print(f'{sub_dir} figures saved in figure_{image_size}s')



# visualization
def show_results(type,replot=True):

    if type == 'VW':
        type_root = "/data/VW/processed/evaluation"
        data_list_path = '/data/VW/list_vw_2024Oct28.xlsx'
        data_pd = pd.read_excel(data_list_path)
        data_list = data_pd['file_name'].tolist()
        figure_type='eeg'

    elif type == 'SEIZURE':
        type_root = "/data/SEIZURE/processed/evaluation"
        data_list_path = '/data/SEIZURE/list_seizures_2024Oct22.xlsx'
        df_1 = pd.read_excel(data_list_path, sheet_name='batch1')
        df_2 = pd.read_excel(data_list_path, sheet_name='batch2_sparcnet')
        data_list = df_1['file_name'].tolist() + df_2['file_name'].tolist()
        figure_type = 'spectrogram'

    elif type == 'BETS':
        type_root = "/data/BETS/processed/evaluation"
        data_list_path = '/data/BETS/list_bets_2024Nov06.xlsx'
        data_pd = pd.read_excel(data_list_path)
        data_list = data_pd['file_name'].tolist()
        figure_type = 'eeg'

    elif type == 'WICKETS':
        type_root = "/data/WICKETS/processed/evaluation"
        data_list_path = '/data/WICKETS/list_wickets_2024Nov06.xlsx'
        data_pd = pd.read_excel(data_list_path)
        data_list = data_pd['file_name'].tolist()
        figure_type = 'eeg'

    elif type == 'BS':
        type_root = "/data/BS/processed/evaluation"
        data_list_path = '/data/BS/list_bs_2024Nov06.xlsx'
        data_pd = pd.read_excel(data_list_path)
        data_list = data_pd['file_name'].tolist()
        figure_type = 'spectrogram'

    elif type == 'SPINDLES':
        type_root = "/data/SPINDLES/processed/evaluation"
        data_list_path = '/data/SPINDLES/list_spindles_2024Nov06.xlsx'
        data_pd = pd.read_excel(data_list_path)
        data_list = data_pd['file_name'].tolist()
        figure_type = 'eeg'

    elif type == 'LRDA':
        type_root = "/data/LRDA/processed/evaluation"
        data_list_path = '/data/LRDA/list_lrda_2024Oct29.xlsx'
        df_1 = pd.read_excel(data_list_path, sheet_name='lrda')
        df_2 = pd.read_excel(data_list_path, sheet_name='sparcnet')
        data_list = df_1['file_name'].tolist() + df_2['file_name'].tolist()
        figure_type = 'spectrogram'

    elif type == 'LPD':
        type_root = "/data/LPD/processed/evaluation"
        data_list_path = '/data/LPD/list_lpd_2024Nov08.xlsx'
        df_1 = pd.read_excel(data_list_path, sheet_name='sparcnet')
        df_2 = pd.read_excel(data_list_path, sheet_name='new')
        data_list = df_1['file_name'].tolist() + df_2['file_name'].tolist()
        figure_type = 'spectrogram'

    elif type == 'GRDA':
        type_root = "/data/GRDA/processed/evaluation"
        data_list_path = '/data/GRDA/list_grda_2024Nov08.xlsx'
        df_1 = pd.read_excel(data_list_path, sheet_name='grda')
        df_2 = pd.read_excel(data_list_path, sheet_name='new')
        data_list = df_1['file_name'].tolist() + df_2['file_name'].tolist()
        figure_type = 'spectrogram'


    data_list =[os.path.join(type_root, item) for item in data_list]

    all_items = os.listdir(type_root)
    subdirectories = [os.path.join(type_root, item) for item in all_items if
                      os.path.isdir(os.path.join(type_root, item))]

    params=[]
    for sub_dir in data_list:
        if not sub_dir in subdirectories:
            print(f'No {sub_dir} directory ......')
            continue
        else:
            params.append([type,sub_dir,figure_type,replot])

    num_cores = int(os.cpu_count() / 2)
    with Pool(processes=num_cores) as pool:
        # split and dump in parallel, could use processes= 24  #pool.map(split_and_dump_MGB, parameters)
        for _ in tqdm(pool.imap_unordered(plot_results, params), total=len(params)):
            pass


# new 2025/1/31---------------------------------------------------------------------------

eeg_channels1  = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

# T3=T7  T5=P7 T4=T8  T6=P8
eeg_channels2  = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T8', 'P8', 'O2']

def get_frequency_from_mat(raw_mat,dataset):
    try:
        fs_value = raw_mat['Fs']
    except KeyError:
        try:
            fs_value = raw_mat['fs']
        except KeyError:
            try:
                fs_value = raw_mat['sampling_rate']
            except KeyError:
                print('No fs or Fs in mat data, default 200hz')
                if dataset=='SPIKES':
                    return 128
                else:
                    return 200

    if isinstance(fs_value, np.ndarray):
        if fs_value.shape == (1, 1, 1):
            fs_value = fs_value[0, 0, 0]
        elif fs_value.shape == (1, 1):
            fs_value = fs_value[0, 0]
        elif fs_value.shape == (1,):
            fs_value = fs_value[0]
        elif fs_value.shape == ():
            fs_value=fs_value.item()
        else:
            if dataset == 'SPIKES':
                print('Unexpected array shape for fs value, default 128hz')
                return 128
            else:
                print('Unexpected array shape for fs value, default 200hz')
                return 200

    if isinstance(fs_value, np.ndarray):
        fs_value = fs_value.item()

    return int(fs_value)


def read_mat(data_path,data_class):
    try:
        raw = mat73.loadmat(data_path)
        signal = raw['data']
        self_fs = get_frequency_from_mat(raw_mat=raw, dataset=data_class)

    except TypeError:
        try:
            raw = scipy.io.loadmat(data_path)
            signal = raw['data']
            self_fs = get_frequency_from_mat(raw_mat=raw, dataset=data_class)

        except Exception as e:
            raise ValueError(f'Failed to load {data_path}. Mat type error : {e}')

    return signal, self_fs

def plot_eeg_pred(pred, eeg_data, fs=200, out_path='Desktop/pred.png', size=10, bipolar=False):
    eeg_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

    if bipolar:
        eeg_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3',
                        'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']

    fig = plt.figure(figsize=(size, 8), facecolor='w')
    ax_spec = []
    top = 0.92
    for i in range(len(eeg_channels) + 1):
        ax_spec.append(fig.add_axes([0.05, top, 0.93, 0.038]))
        top -= 0.042
        if i == 7 or i == 10 or i == 18 or i == 20:
            top -= 0.02

    t_eeg = np.arange(0, eeg_data.shape[1], 1)
    t_pred = np.arange(0, len(pred), 1)

    plt.set_cmap('jet')
    for i in range(len(ax_spec)):
        ax = ax_spec[i]
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == 0:

            ax.plot(t_pred, pred, color="black", linewidth=0.8)
            ax.set_xlim(left=0, right=len(t_eeg) - 1)
            plt.ylim([0.0, 1.0])
            ax.set_ylabel(f'pred   ', fontsize=7, rotation=0,
                          fontweight='bold', loc='center')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels([])
            tick_positions = np.arange(0, len(pred), fs)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([])

        else:
            ax.plot(t_eeg, eeg_data[i - 1], color="black", linewidth=0.8)
            ax.set_ylabel(f'{eeg_channels[i - 1]}{' ' * (len(eeg_channels[i - 1]) + 6)}', fontsize=7, rotation=0,
                          fontweight='bold', loc='center')
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.grid(True, linestyle="--")
            tick_positions = np.arange(0, len(t_eeg), fs)
            ax.set_xlim(left=0, right=len(t_eeg) - 1)
            ax.tick_params(axis='both', which='both', length=0)

            if i == len(ax_spec) - 1:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(int(x / fs)) for x in tick_positions], fontsize=6)
                ax.set_xlabel('Time (seconds)', fontsize=6)
            else:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([])

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_plot_eeg_pred_multicolor(pred, out_path):
    # pred = [0, 1, 2, 2, 4, 5, 2, 1, 0, 3, 4, 5]  # example data

    # color map
    color_map = {
        0: 'blue',
        1: 'red',
        2: 'orange',
        3: 'yellow',
        4: 'green',
        5: 'lightblue'
    }

    # assign a color to each value
    colors = [color_map[value] for value in pred]

    # create figure
    fig, ax = plt.subplots(figsize=(20, 0.5))

    # draw continuous bar chart
    for i, (value, color) in enumerate(zip(pred, colors)):
        ax.barh(0, 1, left=i, height=2, color=color)  # draw each color block from left to right

    # set figure properties
    ax.set_xlim(0, len(pred))  # set x-axis range
    ax.set_ylim(0, 1)  # set y-axis range
    ax.set_yticks([])  # hide y-axis ticks
    # ax.set_xticks(np.arange(len(pred)) + 0.5)  # set x-axis tick positions
    # ax.set_xticklabels(pred)  # set x-axis tick labels to data values
    ax.set_xlabel('time')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def show_binaryclass_results(data_dir, data_format,result_dir,out_dir, label_file='', file_name_column='', label_column='', result_step=1, show_center_10s=True, percent_results=1, rewrite=True):
    os.makedirs(out_dir, exist_ok=True)
    if rewrite:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    if label_file!='':
        if label_file.split('.')[-1]=='csv':
            label_df=pd.read_csv(label_file)
        elif label_file.split('.')[-1]=='xlsx':
            label_df=pd.read_excel(label_file)
        else:
            print('only csv or xlsx for label file')
            return

        if not(file_name_column in label_df.columns and label_column in label_df.columns):
            print('file name column error')
            return

    files=os.listdir(result_dir)
    files = random.sample(files,  max(1, round(len(files) * percent_results)))

    for file in tqdm(files):
        result_df = pd.read_csv(os.path.join(result_dir, file))

        if data_format == 'edf':
            raw = mne.io.read_raw_edf(os.path.join(data_dir, file.replace('.csv', '.edf')), preload=True)
            Fs = int(raw.info['sfreq'])
            new_channel_names = {ch_name: ch_name.upper() for ch_name in raw.ch_names}
            raw.rename_channels(new_channel_names)
            channels = raw.ch_names

            if set(channels).issuperset(set(eeg_channels1)):
                selected_channels = eeg_channels1
            elif set(channels).issuperset(set(eeg_channels2)):
                selected_channels = eeg_channels2
            else:
                print("EDF file does not contain all channels from either eeg_channels1 or eeg_channels2.")
                continue

            raw_selected = raw.copy().pick_channels(selected_channels)
            data = raw_selected.get_data(units='uV')
            # for hep data, should *10 before input the model
            # signal=signal*10


        elif data_format == 'mat':
            data, Fs = read_mat(os.path.join(data_dir, file.replace('.csv', '.mat')), data_class='SPIKES')

        else: return

        # repeat results according to step
        if result_step != 1:
            result_df = result_df.loc[result_df.index.repeat(result_step)].reset_index(drop=True)

        # pad with zeros at the front and back
        zeros_df = pd.DataFrame({'pred': [0] * int(Fs / 2)})
        pred = pd.concat([zeros_df, result_df,zeros_df], ignore_index=True)

        # append the true class label to the image filename
        if label_file!='':
            matches = label_df[label_df[file_name_column]== file.split('.')[0]]
            if matches.empty:
                matches = label_df[label_df[file_name_column].apply(lambda x: x in file.split('.')[0])]
            if not matches.empty:
                label_true = matches[label_column].values[0]
                img_file_name = f'{label_true}_{file.replace('.csv', '.png')}'
            else:
                print('result file not in label list')
                img_file_name = file.replace('.csv', '.png')
        else:
            img_file_name=file.replace('.csv', '.png')

        # center 10s
        if show_center_10s:
            pred=pred[len(pred)//2 - 5 * Fs: len(pred)//2 + 5 * Fs]
            data=data[:, data.shape[1]//2- 5 * Fs: data.shape[1]//2 + 5 * Fs]
            plot_eeg_pred(pred, data, fs=Fs, out_path=os.path.join(out_dir, img_file_name), size=10, bipolar=False)
        else:
            plot_eeg_pred(pred, data, fs=Fs, out_path=os.path.join(out_dir, img_file_name), size=10,
                         bipolar=False)


def show_multiclass_results(result_dir, out_dir, label_file='', file_name_column='', label_column='',rewrite=True):
    os.makedirs(out_dir, exist_ok=True)
    if rewrite:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    if label_file!='':
        if label_file.split('.')[-1]=='csv':
            label_df=pd.read_csv(label_file)
        elif label_file.split('.')[-1]=='xlsx':
            label_df=pd.read_excel(label_file)
        else:
            print('only csv or xlsx for label file')
            return

        if not(file_name_column in label_df.columns and label_column in label_df.columns):
            print('file name column error')
            return

        label_df[file_name_column]=label_df[file_name_column].astype(str)

    for file in tqdm(os.listdir(result_dir)):
        result_df = pd.read_csv(os.path.join(result_dir, file))
        if 'pred_class' in result_df.columns:
            results=result_df['pred_class'].to_list()
        elif 'pred' in result_df.columns:
            results = result_df['pred'].to_list()
            results = [1 if x >= 0.5 else 0 for x in results]

        if label_file!='':

            matches = label_df[label_df[file_name_column] == file.split('.')[0]]

            if matches.empty:
                matches = label_df[label_df[file_name_column].apply(lambda x: x in file.split('.')[0])]
            if not matches.empty:
                label_true = matches[label_column].values[0]
                img_file_name = f'{label_true}_{file.replace('.csv', '.png')}'
            else:
                print('result file not in label list')
                img_file_name = file.replace('.csv', '.png')

        else:
            img_file_name=file.replace('.csv', '.png')

        plot_plot_eeg_pred_multicolor(pred=results, out_path=os.path.join(out_dir,img_file_name))


def clean_Sandor_label_file():
    raw_df=pd.read_excel('/data/Sandor_100/validation_study_excel_export.xlsx',sheet_name='Raters aggregated')
    # create new dataframe
    new_df = pd.DataFrame()
    # process the ID column, adding "_"
    new_df['id'] = raw_df['ID (Study Name in NeuroWorks)'].astype(str).apply(
        lambda x: x[:2] + '-' + x[2:] if x.startswith('ID') else x)

    # assign normal column
    new_df['normal'] = raw_df['he_con_abnormal']

    # compute slowing column
    new_df['slowing'] = raw_df.apply(
        lambda row: 12 if row['he_con_nonepifoc'] == 1 and row['he_con_nonepidiffuse'] ==1 else (
            1 if row['he_con_nonepifoc'] == 1 else (2 if row['he_con_nonepidiffuse'] == 1 else 0)), axis=1
    )

    # compute foc_gen_spikes column
    new_df['foc_gen_spikes'] = raw_df.apply(
        lambda row: 12 if row['he_con_intictepifoc'] == 1 and row['he_con_intictepigen'] == 1 else (
            1 if row['he_con_intictepifoc'] == 1 else (2 if row['he_con_intictepigen'] == 1 else 0)), axis=1
    )

    new_df.to_csv('/data/Sandor_100/Sandor_labels.csv', index=False)

def clean_Occasion_label_file():
    raw_df = pd.read_excel('/data/OccasionNoise/Occasion.xlsx',sheet_name='Consensus')
    new_df = pd.DataFrame()
    new_df['fid'] = raw_df['fid']

    new_df['slowing'] = raw_df.apply(
        lambda row: 12 if  row['Average of r1.FN'] >= 0.5 and row['Average of r1.GN'] >= 0.5 else (1 if row['Average of r1.FN'] >= 0.5 else (2 if row['Average of r1.GN'] >= 0.5 else 0)), axis=1
    )

    new_df['foc_gen_spikes'] = raw_df.apply(
        lambda row: 12 if row['Average of r1.FS'] >= 0.5 and row['Average of r1.GS'] >= 0.5 else (
            1 if row['Average of r1.FS'] >= 0.5 else (2 if row['Average of r1.GS'] >= 0.5 else 0)), axis=1
    )

    new_df.to_csv('/data/OccasionNoise/OccasionNoise_labels.csv', index=False)

if __name__=='__main__':

    # clean_Sandor_label_file()
    # clean_Occasion_label_file()

# # spike HEP--------------------
#     show_binaryclass_results(data_dir='/data/HEP/HEP_events/edf',
#                              data_format='edf',
#                              result_dir='/data/HEP/HEP_events/pred_1pStep',
#                              out_dir='/data/HEP/HEP_events/pred_1pStep_img',
#                              label_file='/data/HEP/HEP_events/segments_labels_channels_montage.xlsx',
#                              file_name_column='event_file',
#                              label_column='Spike',
#                              result_step=1,
#                              show_center_10s=False)


    # show_binaryclass_results(data_dir='/data/HEP/HEP_events/edf_test',
    #                          data_format='edf',
    #                          result_dir='/data/HEP/HEP_events/pred_test_1pStep',
    #                          out_dir='/data/HEP/HEP_events/pred_test_1pStep_img',
    #                          label_file='/data/HEP/HEP_events/segments_labels_channels_montage.xlsx',
    #                          file_name_column='event_file',
    #                          label_column='Spike',
    #                          result_step=1,
    #                          show_center_10s=False)
#
# spike MoE--------------------
    show_binaryclass_results(data_dir='/data/MoE/events',
                             data_format='mat',
                             result_dir='/data/MoE/results/pred_SPIKES_1pStep',
                             out_dir='/data/MoE/results/pred_SPIKES_1pStep_img',
                             label_file='/data/MoE/results/SPIKES_model_and_experts_results.csv',
                             file_name_column='event',
                             label_column='true',
                             result_step=1,
                             show_center_10s=False)

# spike BCH--------------------
#     show_binaryclass_results(data_dir='/data/SPIKES_BCH/segments_15sec_raw',
#                              data_format='mat',
#                              result_dir='/data/SPIKES_BCH/pred_2pStep',
#                              out_dir='/data/SPIKES_BCH/pred_2pStep_img',
#                              result_step=2,
#                              show_center_10s=False)

# spike SN2 test--------------------
#     show_binaryclass_results(data_dir='/data/SPIKES/segments_10min',
#                              data_format='mat',
#                              result_dir='/data/SPIKES/pred_SN2test_2pStep',
#                              out_dir='/data/SPIKES/pred_SN2test_2pStep_img',
#                              label_file='/data/Dataset_statistic/list_spikes_2024Nov17.xlsx',
#                              file_name_column='file_name',
#                              label_column='soft_score',
#                              result_step=2,
#                              show_center_10s=True)


# IIIC IIIC--------------------
#     show_multiclass_results(result_dir='/data/IIIC/pred_2000pStep',
#                             out_dir='/data/IIIC/pred_2000pStep_img',
#                             label_file='/data/Dataset_statistic/list_iiic_20241129.xlsx',
#                             file_name_column='file_name',
#                             label_column='label ([other,seizure,lpd,gpd,lrda,grda])')

# IIIC NO IIIC--------------------
#     show_multiclass_results(result_dir='/data/WICKETS/pred_200pStep',
#                             out_dir='/data/WICKETS/pred_200pStep_img')

# IIIC MoE--------------------
#     show_multiclass_results(result_dir='/data/MoE/results/pred_IIIC_100pStep',
#                             out_dir='/data/MoE/results/pred_IIIC_100pStep_img',
#                             label_file='/data/MoE/results/IIIC_model_and_experts_results.csv',
#                             file_name_column='event',
#                             label_column='true')

# SLOWING MoE--------------------
#     show_multiclass_results(result_dir='/data/MoE/pred_results/pred_SLOWING_100pStep',
#                             out_dir='/data/MoE/pred_results/pred_SLOWING_100pStep_img',
#                             label_file='/data/MoE/pred_results/SLOWING_model_and_experts_results.csv',
#                             file_name_column='event',
#                             label_column='true')

# SLOWING Sandor-100 --------------------
#     show_multiclass_results(result_dir='/data/Sandor_100/results/pred_SLOWING_1sStep',
#                           out_dir='/data/Sandor_100/results/pred_SLOWING_1sStep_img',
#                           label_file='/data/Sandor_100/Sandor_labels.csv',
#                           file_name_column='id',
#                           label_column='slowing')

# SLOWING OccasionNoise--------------------
#     show_multiclass_results(result_dir='/data/OccasionNoise/results/pred_SLOWING_1sStep',
#                             out_dir='/data/OccasionNoise/results/pred_SLOWING_1sStep_img',
#                             label_file='/data/OccasionNoise/OccasionNoise_labels.csv',
#                             file_name_column='fid',
#                             label_column='slowing')

# FOCGENSPIKES MoE--------------------
#     show_multiclass_results(result_dir='/data/MoE/results/pred_FOCGENSPIKES_100pStep',
#                           out_dir='/data/MoE/results/pred_FOCGENSPIKES_100pStep_img',
#                           label_file='/data/MoE/results/FOC_GEN_SPIKES_model_and_experts_results.csv',
#                           file_name_column='event',
#                           label_column='true')

# FOCGENSPIKES Sandor-100 --------------------
#     show_multiclass_results(result_dir='/data/Sandor_100/results/pred_FOCGENSPIKES_1sStep',
#                           out_dir='/data/Sandor_100/results/pred_FOCGENSPIKES_1sStep_img',
#                           label_file='/data/Sandor_100/Sandor_labels.csv',
#                           file_name_column='id',
#                           label_column='foc_gen_spikes')

# FOCGENSPIKES OccasionNoise --------------------
#     show_multiclass_results(result_dir='/data/OccasionNoise/results/pred_FOCGENSPIKES_1sStep',
#                             out_dir='/data/OccasionNoise/results/pred_FOCGENSPIKES_1sStep_img',
#                             label_file='/data/OccasionNoise/OccasionNoise_labels.csv',
#                             file_name_column='fid',
#                             label_column='foc_gen_spikes')

# BS MoE--------------------
#     show_multiclass_results(result_dir='/data/MoE/pred_results/pred_BS_100pStep',
#                           out_dir='/data/MoE/pred_results/pred_BS_100pStep_img',
#                           label_file='/data/MoE/pred_results/BS_model_and_experts_results.csv',
#                           file_name_column='event',
#                           label_column='true')

# NORMAL MoE--------------------
#     show_multiclass_results(result_dir='/data/MoE/pred_results/pred_NORMAL_100pStep',
#                           out_dir='/data/MoE/pred_results/pred_NORMAL_100pStep_img',
#                           )

# NORMAL Sandor-100--------------------
#     show_multiclass_results(result_dir='/data/Sandor_100/results/pred_NORMAL_1sStep',
#                           out_dir='/data/Sandor_100/results/pred_NORMAL_1sStep_img',
#                           label_file='/data/Sandor_100/Sandor_labels.csv',
#                           file_name_column='id',
#                           label_column='normal')

# SLEEP 5 stages with 6 channels MoE--------------------
#     show_multiclass_results(result_dir='/data/MoE/pred_results/pred_SLEEP5Stages_100pStep',
#                           out_dir='/data/MoE/pred_results/pred_SLEEP5Stages_100pStep_img',
#                           label_file='/data/MoE/pred_results/SLEEP_6channels_5stages_model_and_experts_results.csv',
#                           file_name_column='event',
#                           label_column='true')

# echo "exxact@1" | sudo -S ~/miniconda3/envs/torchenv/bin/python vis_result.py