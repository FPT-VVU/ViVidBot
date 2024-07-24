python vividbot/data/task/instruct500k.py --task "translate" \
                                    --file-path "/home/duytran/Downloads/output_ds/instruct500k_vi.json" \
                                    --select -1 \
                                    --batch-size 100 \
                                    --num-shard 10 \
                                    --output-dir "/home/duytran/Downloads/output" \
                                    --cache-dir "/home/duytran/Downloads/cache_dir" \
                                    --num-proc 16

                                    