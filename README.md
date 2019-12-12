# Image Captioning

This project is highly inspired from the tensorflow tutorial on Image Captioning. To build your own model check out the tutorial [here](https://www.tensorflow.org/tutorials/text/image_captioning)

The model was trained using more than half a million captions from MS-COCO dataset.
## Example
![A person riding a bike](https://www.feteduvelomarseille.com/wp-content/uploads/2018/08/Pentes-de-lAlpes-dHuez-960x641.jpg)

```bash
Predicted Caption: a man riding a dirt bike on a dirt path <end>
```
![Image showing the attention of the model](https://github.com/kskd1804/image-captioning/blob/master/image-captioning-output.png?raw=true)
## Usage
- Download the notebook.py and checkpoints folder. 
- Place the target image in the root directory and change the image name in the code (line 370) in notebook.py.
- Open Command Prompt or Terminal and change directory to project root directory.
- Execute below statement.
```bash
python notebook.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
