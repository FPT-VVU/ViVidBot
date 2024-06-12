# python vividbot/data/task/videoinstruck100k.py --task "generate" \
#                                     --file-path "/home/duytran/Downloads/vividbot_data/VideoInstruct_Dataset.json" \
#                                     --select -1 \
#                                     --batch-size 100 \
#                                     --repo-id "Vividbot/videoinstruck100k" \
#                                     --num-shards 10 \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \
#                                     --cache-dir "/home/duytran/Downloads/cache_dir" \
#                                     --name-out "videoinstruck100_vi.json" \
#                                     --num-proc 16 

python vividbot/data/task/videoinstruck100k.py --task "download" \
                                    --file-path "/home/duytran/Downloads/output_ds/videoinstruck_100k_chunk_vi" \
                                    --batch-size 100 \
                                    --repo-id "Vividbot/videoinstruck100k" \
                                    --upload-to-hub \
                                    --num-shards 10 \
                                    --output-dir "/home/duytran/Downloads/output_video" \
                                    --cache-dir "/home/duytran/Downloads/cache_dir" \
                                    --num-proc 16

# python vividbot/data/task/videoinstruck100k.py --task "rename column" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast2M_vi.json" \
#                                     --repo-id "Vividbot/vast27_en" \
#                                     --list-old-name "conversation" \
#                                     --list-new-name "conversations"  \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \
#                                     --clean-input
                                    
# python vividbot/data/task/videoinstruck100k.py --task "remove sample" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast_2m_chunk_vi/shard_0.json" \
#                                     --error-file-path "/home/duytran/Downloads/output_video/error/error_shard_0.json" \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \


# python vividbot/data/task/videoinstruck100k.py --task "divide dataset" \
#                                     --file-path "/home/duytran/Downloads/output_ds/videoinstruck100_vi.json" \
#                                     --output-dir "/home/duytran/Downloads/output_ds/videoinstruck_100k_chunk_vi" \
#                                     --num-shards 100