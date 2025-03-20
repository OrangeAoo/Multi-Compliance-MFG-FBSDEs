# Light Version: 2 Period with Inventory Penalty

Check the following items before running the codes.

1. Python packages have been properly installed. If not, run the cmd:
   ```python 
    pip install -r requirements.txt
   ```
2. Working path: '~/LightPackage'
   
3. Script path: '~/LightPackage/Script-2P-withInvPenalty.py'. Run it using the cmd: 
   1. If you wish not to load the existing models:

    ```python 
    python ./Script-2P-withInvPenalty.py --load False
    ```

    2. Otherwise, specify the model configs by setting `GlobalParams1` and `GlobalParams2` in the script; then run the cmd line:

    ```python 
    python ./Script-2P-withInvPenalty.py --load True
    ```
    **Note:** the device will be automatically detected. Preferably on the cuda:0.

All the best :)