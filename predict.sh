python3 predict.py flowers/valid/102/image_08014.jpg models/checkpoint_vgg16 --gpu --top_k 3  
echo "blackberry lily"

python3 predict.py flowers/test/22/image_05360.jpg models/checkpoint_vgg16 --gpu --top_k 4 
echo "pincushion flower"

python3 predict.py flowers/valid/102/image_08014.jpg models/checkpoint_vgg13 --gpu --top_k 3  
echo "blackberry lily"

python3 predict.py flowers/test/22/image_05360.jpg models/checkpoint_vgg13 --gpu --top_k 4 
echo "pincushion flower"

