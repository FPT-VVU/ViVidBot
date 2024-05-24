python3 vividbot/data/task/vast2M.py --task "download" \
                                    --file-path "/root/workspace/chunks" \
                                    --batch-size 100 \
                                    --repo-id "Vividbot/vast2m_vi" \
                                    --upload-to-hub \
                                    --num-shards 10 \
                                    --output-dir "/root/workspace/ViVidBot/output" \
                                    --cache-dir "/root/workspace/ViVidBot/cache" \
                                    --num-proc 16