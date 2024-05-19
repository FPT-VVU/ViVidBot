# python vividbot/data/task/vast2M.py --task "generate" \
#                                     --file-path "/home/duytran/Downloads/vividbot_data/Vast2M.json" \
#                                     --select 1000 \
#                                     --batch-size 100 \
#                                     --repo-id "Vividbot/vast2m_vi" \
#                                     --upload-to-hub \
#                                     --overwrite \
#                                     --num-shards 10 \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \
#                                     --cache-dir "/home/duytran/Downloads/cache_dir" \
#                                     --num-proc 16 

python vividbot/data/task/vast2M.py --task "download" \
                                    --file-path "/home/duytran/Downloads/vividbot_data/Vast2M.json" \
                                    --select 1000 \
                                    --batch-size 100 \
                                    --repo-id "Vividbot/vast2m_vi" \
                                    --upload-to-hub \
                                    --num-shards 10 \
                                    --output-dir "/home/duytran/Downloads/output" \
                                    --cache-dir "/home/duytran/Downloads/cache_dir" \
                                    --num-proc 16

# python vividbot/data/task/vast2M.py --task "rename column" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast2M_vi.json" \
#                                     --repo-id "Vividbot/vast27_en" \
#                                     --list-old-name "conversation" \
#                                     --list-new-name "conversations"  \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \
#                                     --clean-input
                                    
# python vividbot/data/task/vast2M.py --task "remove sample" \
#                                     --file-path "/home/duytran/Downloads/output_ds/vast2M_vi.json" \
#                                     --error-file-path "/home/duytran/Downloads/output/error.json" \
#                                     --repo-id "Vividbot/vast27_en" \
#                                     --output-dir "/home/duytran/Downloads/output_ds" \
#                                     --clean-input