python vividbot/data/task/instruct500k.py --task "generate" \
                                    --file-path "/home/duytran/Downloads/vividbot_data/blip_laion_cc_sbu_558k.json" \
                                    --select -1 \
                                    --batch-size 100 \
                                    --num-shard 32 \
                                    --output-dir "/home/duytran/Downloads/output" \
                                    --cache-dir "/home/duytran/Downloads/cache_dir" \
                                    --num-proc 16
                                    