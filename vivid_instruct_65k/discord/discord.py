import requests


class DiscordNotif:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send(self, body):
        requests.post(self.webhook_url, json=body)


# const data = await fetch(USER_MESSAGE_DISCORD_WEBHOOK_ENDPOINT, {
#     method: "POST",
#     headers,
#     body: JSON.stringify({
#       embeds: [
#         {
#           title: "💬 New Message for Formularizer",
#           description: message,
#           color: 2278494,
#           fields: [
#             {
#               name: "Name",
#               value: user?.name,
#             },
#             {
#               name: "Email",
#               value: user?.email,
#             },
#             {
#               name: "Subscription Status",
#               value: user?.subscription?.status ?? "none",
#             },
#             {
#               name: "Reaction",
#               value:
#                 reaction === "neutral"
#                   ? "😐"
#                   : reaction === "negative"
#                     ? "👎"
#                     : "👍",
#             },
#           ],
#           author: {
#             name: user?.name,
#             icon_url: user?.avatar,
#           },
#           timestamp: new Date().toISOString(),
#         },
#       ],
#     }),
#   })