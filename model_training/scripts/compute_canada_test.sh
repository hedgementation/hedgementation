SOURCEDIR=/home/nathan/Desktop/model_training

python3 $SOURCEDIR/train.py\
    --normalization=minmax \
    --bucketization=uniform \
    --num_buckets=4 \
    --num_epochs=1 \
    --weighted \
    --data_path=data \
    --save_path=$SOURCEDIR/models \