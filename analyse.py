import wandb
import json
import pandas as pd
import matplotlib.pyplot as plt

#### Params

run_name = 'obgmafe9' ## correct labels
max_epoch = 220

run_name = '85sspbew' ## random labels
max_epoch = 220

#### Get data

run = wandb.init()
list = []
for epoch in range(0, max_epoch):

    artifact = run.use_artifact('nazderaze/pytorch-image-models/run-'+run_name+f'-watch_log:v{epoch}', type='run_table')
    with open(f"{artifact.download()}/watch_log.table.json") as file:
        json_dict = json.load(file)
    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
    df['epoch'] = epoch
    list.append(df)

data = pd.concat(list, ignore_index=True)

data['correct_pred'] = (data['target'] == data['pred']).astype(int)
data = data.sort_values(by=['hash', 'epoch'])
data['correct_pred_previous'] = data.groupby('hash')['correct_pred'].shift()

accuracy = data.groupby('epoch')['correct_pred'].mean()*100
cond_accuracy = data.loc[data['correct_pred_previous'] == 1, ['epoch', 'correct_pred']].groupby('epoch').mean()*100
n_unique = data.groupby('epoch')['pred'].nunique()

# Plotting
plt.plot(accuracy.index, accuracy.values, color='blue')
plt.plot(cond_accuracy.index, cond_accuracy.values, color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

#### Store data

import wandb
import pandas as pd
import json

runs_dict = {
    'xs40qb6m': 16,
    'ej3jjn1i': 24,
    'lmamhc8y': 32,
    '1hdxsz0l': 40,
    'm02titzv': 48,
    '4p1t1bfq': 56,
    'z9gwbob0': 64
}

for run_name, channels in runs_dict.items():

    run = wandb.init()
    list = []
    epoch = 0
    while True:
        try:
            artifact = run.use_artifact('nazderaze/pytorch-image-models/run-' + run_name + f'-watch_log:v{epoch}',
                                        type='run_table')
            with open(f"{artifact.download()}/watch_log.table.json") as file:
                json_dict = json.load(file)
            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
            df['epoch'] = epoch
            list.append(df)
        except Exception as e:
            print(f"Channels: {channels}. Epochs: {epoch}")
            break
        epoch += 1
    data = pd.concat(list, ignore_index=True)

    data['correct_pred'] = (data['target'] == data['pred']).astype(int)
    data = data.sort_values(by=['hash', 'epoch'])
    data['correct_pred_previous'] = data.groupby('hash')['correct_pred'].shift()

    accuracy = pd.DataFrame(data.groupby('epoch')['correct_pred'].mean() * 100).reset_index()
    cond_accuracy = (data.loc[data['correct_pred_previous'] == 1, ['epoch', 'correct_pred']].groupby(
        'epoch').mean() * 100).reset_index()

    to_save = pd.merge(accuracy, cond_accuracy, on='epoch', suffixes=['', '_cond'])
    to_save.to_csv(f'output/resnet50-{channels}.csv')
    run.finish()


#### Plot all
import pandas as pd
import matplotlib.pyplot as plt

dataframes = []
for channels in [16, 24, 32, 40, 48, 56, 64]:
    df_temp = pd.read_csv(f'output/resnet50-{channels}.csv')
    df_temp['channels'] = channels
    dataframes.append(df_temp)
data = pd.concat(dataframes, ignore_index=True)

# train-top1
fig,ax = plt.subplots()
for k,v in data.groupby('channels'):
    v.plot(x='epoch', y='correct_pred', ax=ax, label=k)
plt.xlabel('Epoch')
plt.ylabel('train-top1')
plt.title('Random target on resnet50')
plt.legend(title='Channels')
plt.grid(True)
plt.show()

# train-top1-cond
fig,ax = plt.subplots()
for k,v in data.groupby('channels'):
    v.plot(x='epoch', y='correct_pred_cond', ax=ax, label=k)
plt.xlabel('Epoch')
plt.ylabel('train-top1')
plt.title('Random target on resnet50')
plt.legend(title='Channels')
plt.grid(True)
plt.show()