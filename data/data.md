# Data Setup

## Option A — MIMIC-CXR (requires PhysioNet account, free)
1. Go to https://physionet.org/content/mimic-cxr-jpg/
2. Sign up, complete CITI training, get approved (takes ~1 day)
3. wget -r -N -c -np --user YOUR_USER --ask-password \
   https://physionet.org/files/mimic-cxr-jpg/2.0.0/

## Option B — IU-Xray (public, no approval needed)
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
tar -xzf NLMCXR_png.tgz -C data/iu-xray/
tar -xzf NLMCXR_reports.tgz -C data/iu-xray/

## Option C — Use the demo dummy data (for running RIGHT NOW)
python data/dummy.py