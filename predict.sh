python3 predict.py flowers/valid/13/image_05772.jpg models/checkpoint_vgg16 --gpu --top_k 3
echo "king protea"

python3 predict.py flowers/test/22/image_05360.jpg models/checkpoint_vgg16 --gpu --top_k 4
echo "pincushion flower"

