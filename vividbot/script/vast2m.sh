python vividbot/data/task/vast2M.py --task "generate" \
                                    --file-path "/home/duytran/Downloads/output_ds/vast_2m_chunk_en" \
                                    --select -1 \
                                    --batch-size 100 \
                                    --repo-id "Vividbot/vast2m_vi" \
                                    --num-shards 10 \
                                    --output-dir "/home/duytran/Downloads/output_ds/vast_2m_chunk_en2vi" \
                                    --cache-dir "/home/duytran/Downloads/cache_dir" \
                                    --num-proc 16 

# python3 vividbot/data/task/vast2M.py --task "download" \
#                                     --file-path "/home/duytran/Downloads/output_ds/chunk_200" \
#                                     --batch-size 100 \
#                                     --repo-id "Vividbot/vast2m_vi" \
#                                     --upload-to-hub \
#                                     --num-shards 10 \
#                                     --output-dir "/home/duytran/Downloads/output_video" \
#                                     --cache-dir "/home/duytran/Downloads/cache_dir" \
#                                     --num-proc 16

# python vividbot/data/task/vast2M.py --task "rename column" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast2M_vi.json" \
#                                     --repo-id "Vividbot/vast27_en" \
#                                     --list-old-name "conversation" \
#                                     --list-new-name "conversations"  \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \
#                                     --clean-input
                                    
# python vividbot/data/task/vast2M.py --task "remove sample" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast_2m_chunk_vi/shard_0.json" \
#                                     --error-file-path "/home/duytran/Downloads/output_video/error/error_shard_0.json" \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \


# python vividbot/data/task/vast2M.py --task "divide dataset" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast2M_vi.json" \
#                                     --output-dir "/home/duytran/Downloads/output_ds/vast_2m_chunk_vi" \
#                                     --num-shards 500