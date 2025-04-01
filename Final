{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8975d55f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:06:55.435705Z",
     "iopub.status.busy": "2024-11-11T14:06:55.435414Z",
     "iopub.status.idle": "2024-11-11T14:07:04.329625Z",
     "shell.execute_reply": "2024-11-11T14:07:04.328795Z"
    },
    "papermill": {
     "duration": 8.904316,
     "end_time": "2024-11-11T14:07:04.331962",
     "exception": false,
     "start_time": "2024-11-11T14:06:55.427646",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import cv2\n",
    "import librosa\n",
    "import subprocess\n",
    "from transformers import BertTokenizer, BertModel\n",
    "import torch\n",
    "import nltk\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "918421b2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:04.346412Z",
     "iopub.status.busy": "2024-11-11T14:07:04.345904Z",
     "iopub.status.idle": "2024-11-11T14:07:04.373213Z",
     "shell.execute_reply": "2024-11-11T14:07:04.372091Z"
    },
    "papermill": {
     "duration": 0.03652,
     "end_time": "2024-11-11T14:07:04.375270",
     "exception": false,
     "start_time": "2024-11-11T14:07:04.338750",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading dataset...\n",
      "Dataset loaded.\n"
     ]
    }
   ],
   "source": [
    "print(\"Loading dataset...\")\n",
    "train_df = pd.read_csv(f'/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_train/train_emotion.csv', encoding='ISO-8859-1')\n",
    "print(\"Dataset loaded.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "66953d25",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:04.388863Z",
     "iopub.status.busy": "2024-11-11T14:07:04.388570Z",
     "iopub.status.idle": "2024-11-11T14:07:04.392524Z",
     "shell.execute_reply": "2024-11-11T14:07:04.391667Z"
    },
    "papermill": {
     "duration": 0.013036,
     "end_time": "2024-11-11T14:07:04.394529",
     "exception": false,
     "start_time": "2024-11-11T14:07:04.381493",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "video_dir = f'/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_train/train_data'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0bceb6c8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:04.409074Z",
     "iopub.status.busy": "2024-11-11T14:07:04.408749Z",
     "iopub.status.idle": "2024-11-11T14:07:04.414288Z",
     "shell.execute_reply": "2024-11-11T14:07:04.413389Z"
    },
    "papermill": {
     "duration": 0.014789,
     "end_time": "2024-11-11T14:07:04.416427",
     "exception": false,
     "start_time": "2024-11-11T14:07:04.401638",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "def get_video_clip_path(row):\n",
    "    dialogue_id = row['Dialogue_ID']\n",
    "    utterance_id = row['Utterance_ID']\n",
    "    filename = f\"dia{dialogue_id}_utt{utterance_id}.mp4\"\n",
    "    return os.path.join(video_dir, filename)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2c5b6b82",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:04.430437Z",
     "iopub.status.busy": "2024-11-11T14:07:04.430147Z",
     "iopub.status.idle": "2024-11-11T14:07:04.458232Z",
     "shell.execute_reply": "2024-11-11T14:07:04.457355Z"
    },
    "papermill": {
     "duration": 0.037332,
     "end_time": "2024-11-11T14:07:04.460098",
     "exception": false,
     "start_time": "2024-11-11T14:07:04.422766",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "train_df['video_clip_path'] = train_df.apply(get_video_clip_path, axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fe7bd950",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:04.474106Z",
     "iopub.status.busy": "2024-11-11T14:07:04.473354Z",
     "iopub.status.idle": "2024-11-11T14:07:08.143005Z",
     "shell.execute_reply": "2024-11-11T14:07:08.142230Z"
    },
    "papermill": {
     "duration": 3.678848,
     "end_time": "2024-11-11T14:07:08.145321",
     "exception": false,
     "start_time": "2024-11-11T14:07:04.466473",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9636a0e903744c50a27fa1fa8de5ae4a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "20d664a4762c46e89748668a878e5eba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "baa7624a234140b9a2a155ea37075303",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c538d52f690d4b3fbd709487c8fe824c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "44e66ae03e9b42fd9559ba05d961473c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')\n",
    "Bmodel = BertModel.from_pretrained('bert-base-uncased')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "347ce655",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:08.161414Z",
     "iopub.status.busy": "2024-11-11T14:07:08.161101Z",
     "iopub.status.idle": "2024-11-11T14:07:08.166435Z",
     "shell.execute_reply": "2024-11-11T14:07:08.165712Z"
    },
    "papermill": {
     "duration": 0.015452,
     "end_time": "2024-11-11T14:07:08.168259",
     "exception": false,
     "start_time": "2024-11-11T14:07:08.152807",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "def extract_bert_embeddings(text):\n",
    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)\n",
    "    with torch.no_grad():\n",
    "        outputs = Bmodel(**inputs)\n",
    "    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()\n",
    "    return cls_embedding\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b69aff83",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:07:08.183318Z",
     "iopub.status.busy": "2024-11-11T14:07:08.183024Z",
     "iopub.status.idle": "2024-11-11T14:08:02.963718Z",
     "shell.execute_reply": "2024-11-11T14:08:02.962757Z"
    },
    "papermill": {
     "duration": 54.797769,
     "end_time": "2024-11-11T14:08:02.972971",
     "exception": false,
     "start_time": "2024-11-11T14:07:08.175202",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting text features using BERT...\n",
      "Text features extracted.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(\"Extracting text features using BERT...\")\n",
    "train_df['text_features'] = train_df['Utterance'].apply(extract_bert_embeddings)\n",
    "print(\"Text features extracted.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e2e68ce4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:08:02.988747Z",
     "iopub.status.busy": "2024-11-11T14:08:02.988446Z",
     "iopub.status.idle": "2024-11-11T14:08:02.993595Z",
     "shell.execute_reply": "2024-11-11T14:08:02.992764Z"
    },
    "papermill": {
     "duration": 0.015303,
     "end_time": "2024-11-11T14:08:02.995476",
     "exception": false,
     "start_time": "2024-11-11T14:08:02.980173",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "def extract_audio_from_video(video_path, audio_path=\"/tmp/temp_audio.wav\"):\n",
    "    command = [\n",
    "        \"ffmpeg\", \n",
    "        \"-i\", video_path,\n",
    "        \"-vn\",\n",
    "        \"-ac\", \"1\",\n",
    "        \"-ar\", \"16000\",\n",
    "        \"-acodec\", \"pcm_s16le\",\n",
    "        audio_path\n",
    "    ]\n",
    "    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f9bd5cca",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:08:03.010672Z",
     "iopub.status.busy": "2024-11-11T14:08:03.010399Z",
     "iopub.status.idle": "2024-11-11T14:08:03.015361Z",
     "shell.execute_reply": "2024-11-11T14:08:03.014573Z"
    },
    "papermill": {
     "duration": 0.014734,
     "end_time": "2024-11-11T14:08:03.017238",
     "exception": false,
     "start_time": "2024-11-11T14:08:03.002504",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def extract_audio_features(video_path):\n",
    "    audio_path = \"/tmp/temp_audio.wav\"\n",
    "    extract_audio_from_video(video_path, audio_path)\n",
    "    y, sr = librosa.load(audio_path, sr=None)\n",
    "    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)\n",
    "    return np.mean(mfccs, axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "930294ee",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:08:03.032716Z",
     "iopub.status.busy": "2024-11-11T14:08:03.032449Z",
     "iopub.status.idle": "2024-11-11T14:11:16.326529Z",
     "shell.execute_reply": "2024-11-11T14:11:16.325221Z"
    },
    "papermill": {
     "duration": 193.306626,
     "end_time": "2024-11-11T14:11:16.330925",
     "exception": false,
     "start_time": "2024-11-11T14:08:03.024299",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting audio features...\n",
      "Audio features extracted.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(\"Extracting audio features...\")\n",
    "train_df['audio_features'] = train_df['video_clip_path'].apply(extract_audio_features)\n",
    "print(\"Audio features extracted.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f02db2e4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:16.366765Z",
     "iopub.status.busy": "2024-11-11T14:11:16.365892Z",
     "iopub.status.idle": "2024-11-11T14:11:16.386595Z",
     "shell.execute_reply": "2024-11-11T14:11:16.385467Z"
    },
    "papermill": {
     "duration": 0.040683,
     "end_time": "2024-11-11T14:11:16.390343",
     "exception": false,
     "start_time": "2024-11-11T14:11:16.349660",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Combining features...\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(\"Combining features...\")\n",
    "combined_features = np.concatenate(( \n",
    "    np.array(train_df['text_features'].tolist()),\n",
    "    np.array(train_df['audio_features'].tolist())\n",
    "), axis=1)\n",
    "\n",
    "X = combined_features\n",
    "y = train_df['Emotion']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c15fcc7e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:16.422166Z",
     "iopub.status.busy": "2024-11-11T14:11:16.421683Z",
     "iopub.status.idle": "2024-11-11T14:11:18.009063Z",
     "shell.execute_reply": "2024-11-11T14:11:18.008155Z"
    },
    "papermill": {
     "duration": 1.606831,
     "end_time": "2024-11-11T14:11:18.012341",
     "exception": false,
     "start_time": "2024-11-11T14:11:16.405510",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.6\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.67      0.08      0.14        25\n",
      "           1       0.35      0.18      0.24        34\n",
      "           2       0.62      0.96      0.75       108\n",
      "           3       0.00      0.00      0.00        13\n",
      "           4       0.67      0.40      0.50        20\n",
      "\n",
      "    accuracy                           0.60       200\n",
      "   macro avg       0.46      0.32      0.33       200\n",
      "weighted avg       0.54      0.60      0.51       200\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "from sklearn.ensemble import RandomForestClassifier  # Example model\n",
    "\n",
    "# Example data\n",
    "# y = ['anger', 'joy', 'neutral', 'sadness', 'surprise']  # Your target labels\n",
    "\n",
    "# Step 1: Label encode the target variable\n",
    "label_encoder = LabelEncoder()\n",
    "y_encoded = label_encoder.fit_transform(y)  # y is your target variable with string labels\n",
    "\n",
    "# Split the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)\n",
    "\n",
    "# Step 2: Train the model\n",
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "# Step 3: Make predictions\n",
    "y_pred = rf_model.predict(X_test)\n",
    "\n",
    "# Step 4: Convert the predicted labels back to original class names\n",
    "y_pred_labels = label_encoder.inverse_transform(y_pred)\n",
    "\n",
    "# Step 5: Print the results\n",
    "print(f\"Accuracy: {accuracy_score(y_test, y_pred)}\")\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# If you want to print confusion matrix or other metrics, you can proceed similarly.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73ff8f61",
   "metadata": {
    "papermill": {
     "duration": 0.007676,
     "end_time": "2024-11-11T14:11:18.027827",
     "exception": false,
     "start_time": "2024-11-11T14:11:18.020151",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5366c860",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:18.044347Z",
     "iopub.status.busy": "2024-11-11T14:11:18.044044Z",
     "iopub.status.idle": "2024-11-11T14:11:34.042603Z",
     "shell.execute_reply": "2024-11-11T14:11:34.041670Z"
    },
    "papermill": {
     "duration": 16.009779,
     "end_time": "2024-11-11T14:11:34.045385",
     "exception": false,
     "start_time": "2024-11-11T14:11:18.035606",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.625\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.62      0.20      0.30        25\n",
      "           1       0.48      0.32      0.39        34\n",
      "           2       0.69      0.94      0.79       108\n",
      "           3       1.00      0.08      0.14        13\n",
      "           4       0.33      0.35      0.34        20\n",
      "\n",
      "    accuracy                           0.62       200\n",
      "   macro avg       0.62      0.38      0.39       200\n",
      "weighted avg       0.63      0.62      0.57       200\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import xgboost as xgb\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "\n",
    "xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)\n",
    "xgb_model.fit(X_train, y_train)\n",
    "\n",
    "y_pred = xgb_model.predict(X_test)\n",
    "print(f\"Accuracy: {accuracy_score(y_test, y_pred)}\")\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "eb7aaf60",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:34.062810Z",
     "iopub.status.busy": "2024-11-11T14:11:34.062520Z",
     "iopub.status.idle": "2024-11-11T14:11:34.072554Z",
     "shell.execute_reply": "2024-11-11T14:11:34.071755Z"
    },
    "papermill": {
     "duration": 0.021117,
     "end_time": "2024-11-11T14:11:34.074575",
     "exception": false,
     "start_time": "2024-11-11T14:11:34.053458",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "test_df = pd.read_csv('/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_test/test_emotion.csv', encoding='ISO-8859-1')\n",
    "test_video_dir = '/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_test/test_data'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1762ba29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:34.092453Z",
     "iopub.status.busy": "2024-11-11T14:11:34.092181Z",
     "iopub.status.idle": "2024-11-11T14:11:34.099261Z",
     "shell.execute_reply": "2024-11-11T14:11:34.098454Z"
    },
    "papermill": {
     "duration": 0.018224,
     "end_time": "2024-11-11T14:11:34.101070",
     "exception": false,
     "start_time": "2024-11-11T14:11:34.082846",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "test_df['video_clip_path'] = test_df.apply(get_video_clip_path, axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "08adf723",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:34.117615Z",
     "iopub.status.busy": "2024-11-11T14:11:34.117307Z",
     "iopub.status.idle": "2024-11-11T14:11:54.893638Z",
     "shell.execute_reply": "2024-11-11T14:11:54.892602Z"
    },
    "papermill": {
     "duration": 20.787191,
     "end_time": "2024-11-11T14:11:54.896089",
     "exception": false,
     "start_time": "2024-11-11T14:11:34.108898",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "test_df['audio_features'] = test_df['video_clip_path'].apply(extract_audio_features)\n",
    "test_df['text_features'] = test_df['Utterance'].apply(extract_bert_embeddings)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e80c71cd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:54.913369Z",
     "iopub.status.busy": "2024-11-11T14:11:54.913053Z",
     "iopub.status.idle": "2024-11-11T14:11:54.918506Z",
     "shell.execute_reply": "2024-11-11T14:11:54.917563Z"
    },
    "papermill": {
     "duration": 0.016428,
     "end_time": "2024-11-11T14:11:54.920669",
     "exception": false,
     "start_time": "2024-11-11T14:11:54.904241",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of individual feature arrays:\n",
      "Audio Features Shape (first row): (13,)\n",
      "Text Features Shape (first row): (768,)\n"
     ]
    }
   ],
   "source": [
    "print(\"Shape of individual feature arrays:\")\n",
    "print(f\"Audio Features Shape (first row): {test_df['audio_features'].iloc[0].shape}\")\n",
    "print(f\"Text Features Shape (first row): {test_df['text_features'].iloc[0].shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b00bf19c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:54.937275Z",
     "iopub.status.busy": "2024-11-11T14:11:54.936972Z",
     "iopub.status.idle": "2024-11-11T14:11:54.943416Z",
     "shell.execute_reply": "2024-11-11T14:11:54.942453Z"
    },
    "papermill": {
     "duration": 0.016997,
     "end_time": "2024-11-11T14:11:54.945430",
     "exception": false,
     "start_time": "2024-11-11T14:11:54.928433",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of combined features (first row): (781,)\n",
      "Shape of combined features (test set): (100, 781)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "combined_features = np.concatenate(( \n",
    "    np.array(test_df['text_features'].tolist()),\n",
    "    np.array(test_df['audio_features'].tolist())\n",
    "), axis=1)\n",
    "\n",
    "print(f\"Shape of combined features (first row): {combined_features[0].shape}\")\n",
    "print(f\"Shape of combined features (test set): {combined_features.shape}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "51612064",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:54.962029Z",
     "iopub.status.busy": "2024-11-11T14:11:54.961705Z",
     "iopub.status.idle": "2024-11-11T14:11:54.967987Z",
     "shell.execute_reply": "2024-11-11T14:11:54.967209Z"
    },
    "papermill": {
     "duration": 0.016653,
     "end_time": "2024-11-11T14:11:54.969888",
     "exception": false,
     "start_time": "2024-11-11T14:11:54.953235",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "predictions = xgb_model.predict(combined_features)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "f520a5dc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-11T14:11:54.986626Z",
     "iopub.status.busy": "2024-11-11T14:11:54.986119Z",
     "iopub.status.idle": "2024-11-11T14:11:54.995530Z",
     "shell.execute_reply": "2024-11-11T14:11:54.994538Z"
    },
    "papermill": {
     "duration": 0.02018,
     "end_time": "2024-11-11T14:11:54.997761",
     "exception": false,
     "start_time": "2024-11-11T14:11:54.977581",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Submission file created.\n"
     ]
    }
   ],
   "source": [
    "# Create the submission DataFrame\n",
    "submission_df = pd.DataFrame({\n",
    "    'Sr No.': test_df['Sr No.'],\n",
    "    'Emotion':  label_encoder.inverse_transform(predictions)\n",
    "\n",
    "})\n",
    "\n",
    "\n",
    "# Save the submission file\n",
    "submission_df.to_csv('submission.csv', index=False)\n",
    "print(\"Submission file created.\")\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "databundleVersionId": 10112483,
     "sourceId": 88318,
     "sourceType": "competition"
    }
   ],
   "dockerImageVersionId": 30786,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 304.968724,
   "end_time": "2024-11-11T14:11:57.471158",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-11-11T14:06:52.502434",
   "version": "2.6.0"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {
     "01adf3298389492fa794bfc637fee52e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "0681a463bf874bd39c19bf2cc3fbc9e1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_e2b16b8b435f4abaa6be77d7964a6e64",
       "max": 48.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_385694a5865b47b4b44e15eb5629455a",
       "value": 48.0
      }
     },
     "0833562042f94a9f892ee04d24a84f5e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "0be28bb90f434bf28874329733580a96": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "0c3488d1c49641bc8c961b8cbda14be3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_58d847107f6a41dc9d62a3ca6c94a29e",
       "max": 440449768.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_cede7b7759c9472d8e300531ab82526d",
       "value": 440449768.0
      }
     },
     "0d5edb4cbd5041ffb5097010d4ad371e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "12fb8854c8014f418c2025cf903b3574": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_a9ebe8fa63d04b408c8e195ec51707c8",
       "placeholder": "​",
       "style": "IPY_MODEL_a7fd5a524181447eb47c1138c1decfdd",
       "value": "vocab.txt: 100%"
      }
     },
     "1487de63942e488a9259692ee1fbc882": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "14a0456747f345838653ef383202221b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_bebc9d4d604c4a5db8f88216163e106a",
       "max": 231508.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_b750fec4adf2440981352e8733e2351d",
       "value": 231508.0
      }
     },
     "184520f635b64cc292c59a9fd85b3888": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "20d664a4762c46e89748668a878e5eba": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_12fb8854c8014f418c2025cf903b3574",
        "IPY_MODEL_14a0456747f345838653ef383202221b",
        "IPY_MODEL_a5b73587b9634c318b17d8fc7f7e956d"
       ],
       "layout": "IPY_MODEL_248550b90c6f4eb8888dfd5a1ff73f78"
      }
     },
     "230ae3c5bcf640528b03051d91b9dee1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_9ae8c268470a4940bec08f5d48b651a4",
       "max": 466062.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_87d422c1eeae4069b14dd83891e4685e",
       "value": 466062.0
      }
     },
     "248550b90c6f4eb8888dfd5a1ff73f78": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "24aaf6dcf8bd46418e38cb6dd51300aa": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "2ba987d104194986a5bf4d6f27eb87db": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "32dda36d8af84269aeca919083dfd375": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_faf359160fb947f3835176261624d60d",
       "placeholder": "​",
       "style": "IPY_MODEL_ea463dde246243dc9e16cdcb41d65d16",
       "value": " 466k/466k [00:00&lt;00:00, 25.8MB/s]"
      }
     },
     "385694a5865b47b4b44e15eb5629455a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "38db851d0f36434584f220a6288dd622": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_7331fec247d349aebad311081bcc4e76",
       "placeholder": "​",
       "style": "IPY_MODEL_9fdde1d70f9e4fca9025cd35027b6824",
       "value": "model.safetensors: 100%"
      }
     },
     "3eb76916b880473da8d887c9e0df42b6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "44e66ae03e9b42fd9559ba05d961473c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_38db851d0f36434584f220a6288dd622",
        "IPY_MODEL_0c3488d1c49641bc8c961b8cbda14be3",
        "IPY_MODEL_55c25405a6ff435b8e651513510ba02b"
       ],
       "layout": "IPY_MODEL_d34d31daca9d46e6a642d060917519f5"
      }
     },
     "55c25405a6ff435b8e651513510ba02b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_baa0474d2e634fe380e18ea661a243a8",
       "placeholder": "​",
       "style": "IPY_MODEL_0be28bb90f434bf28874329733580a96",
       "value": " 440M/440M [00:02&lt;00:00, 245MB/s]"
      }
     },
     "58d847107f6a41dc9d62a3ca6c94a29e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "6e65e8e57f454a56be49c1f22a8e5732": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "7331fec247d349aebad311081bcc4e76": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "812fc9d49a284f548e3dcdb407ffc7ca": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_b51ee37450b841398743b498f1431063",
       "placeholder": "​",
       "style": "IPY_MODEL_c3b48b3f3db54861bd8c792d4b72d221",
       "value": "tokenizer.json: 100%"
      }
     },
     "87bd23cb86f642ca9f25e099a46c335d": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "87d422c1eeae4069b14dd83891e4685e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "8c722a7cbc184ac49be5f4a4265632eb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "9636a0e903744c50a27fa1fa8de5ae4a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_c1a6718ed0e543f6a205f97697b05cc9",
        "IPY_MODEL_0681a463bf874bd39c19bf2cc3fbc9e1",
        "IPY_MODEL_c74c72c3d8ec445ab3730c10d28fd716"
       ],
       "layout": "IPY_MODEL_1487de63942e488a9259692ee1fbc882"
      }
     },
     "9ae8c268470a4940bec08f5d48b651a4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "9fdde1d70f9e4fca9025cd35027b6824": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "a5b73587b9634c318b17d8fc7f7e956d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_01adf3298389492fa794bfc637fee52e",
       "placeholder": "​",
       "style": "IPY_MODEL_0833562042f94a9f892ee04d24a84f5e",
       "value": " 232k/232k [00:00&lt;00:00, 6.73MB/s]"
      }
     },
     "a7fd5a524181447eb47c1138c1decfdd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "a9ebe8fa63d04b408c8e195ec51707c8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "b51ee37450b841398743b498f1431063": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "b750fec4adf2440981352e8733e2351d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "baa0474d2e634fe380e18ea661a243a8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "baa7624a234140b9a2a155ea37075303": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_812fc9d49a284f548e3dcdb407ffc7ca",
        "IPY_MODEL_230ae3c5bcf640528b03051d91b9dee1",
        "IPY_MODEL_32dda36d8af84269aeca919083dfd375"
       ],
       "layout": "IPY_MODEL_87bd23cb86f642ca9f25e099a46c335d"
      }
     },
     "bebc9d4d604c4a5db8f88216163e106a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "c1a6718ed0e543f6a205f97697b05cc9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_24aaf6dcf8bd46418e38cb6dd51300aa",
       "placeholder": "​",
       "style": "IPY_MODEL_fcd871e0bcd845e392c94a543673b422",
       "value": "tokenizer_config.json: 100%"
      }
     },
     "c2cb15443a6c4363a97dea6854937ee6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "c3b48b3f3db54861bd8c792d4b72d221": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "c538d52f690d4b3fbd709487c8fe824c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_f21ca681dc514410886a0457218514fb",
        "IPY_MODEL_f1988f1cadd3439b98dc580e76257be1",
        "IPY_MODEL_f75feee3de58407fbd9a2f4f54e5180b"
       ],
       "layout": "IPY_MODEL_6e65e8e57f454a56be49c1f22a8e5732"
      }
     },
     "c74c72c3d8ec445ab3730c10d28fd716": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_184520f635b64cc292c59a9fd85b3888",
       "placeholder": "​",
       "style": "IPY_MODEL_8c722a7cbc184ac49be5f4a4265632eb",
       "value": " 48.0/48.0 [00:00&lt;00:00, 3.65kB/s]"
      }
     },
     "cede7b7759c9472d8e300531ab82526d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "d34d31daca9d46e6a642d060917519f5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "d75dc0fbd8e441abaca9d063ee0d14c8": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "e2b16b8b435f4abaa6be77d7964a6e64": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "ea463dde246243dc9e16cdcb41d65d16": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "f1988f1cadd3439b98dc580e76257be1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_c2cb15443a6c4363a97dea6854937ee6",
       "max": 570.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_2ba987d104194986a5bf4d6f27eb87db",
       "value": 570.0
      }
     },
     "f21ca681dc514410886a0457218514fb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_0d5edb4cbd5041ffb5097010d4ad371e",
       "placeholder": "​",
       "style": "IPY_MODEL_3eb76916b880473da8d887c9e0df42b6",
       "value": "config.json: 100%"
      }
     },
     "f75feee3de58407fbd9a2f4f54e5180b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_fa00c9506a144db788a3e0a5088f93e9",
       "placeholder": "​",
       "style": "IPY_MODEL_d75dc0fbd8e441abaca9d063ee0d14c8",
       "value": " 570/570 [00:00&lt;00:00, 49.6kB/s]"
      }
     },
     "fa00c9506a144db788a3e0a5088f93e9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "faf359160fb947f3835176261624d60d": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "fcd871e0bcd845e392c94a543673b422": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     }
    },
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
