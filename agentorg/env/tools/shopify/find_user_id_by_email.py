from typing import Any, Dict
import json

import shopify

from agentorg.env.tools.tools import register_tool

description = "Find user id by email. If the user is not found, the function will return an error message."
slots = [
    {
        "name": "email",
        "type": "string",
        "description": "The email of the user.",
        "prompt": "In order to proceed, please provide the email for identity verification.",
        "required": True,
    }
]
outputs = [
    {
        "name": "user_id",
        "type": "string",
        "description": "The user id of the user. such as 'gid://shopify/Customer/13573257450893'.",
    }
]

@register_tool(description, slots, outputs)
def find_user_id_by_email(email: str, **kwargs) -> str:
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return "error: missing some or all required shopify authentication parameters: shop_url, api_version, token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"
    
    user_id = ""
    try:
        with shopify.Session.temp(shop_url, api_version, token):
            response = shopify.GraphQL().execute(f"""
                {{
                    customers (first: 10, query: "email:{email}") {{
                        edges {{
                            node {{
                                id
                            }}
                        }}
                    }}
                }}
                """)
        nodes = json.loads(response)["data"]["customers"]["edges"]
        if len(nodes) == 1:
            user_id = nodes[0]["node"]["id"]
            return user_id
        else:
            return "error: there are multiple users with the same email"
    except Exception as e:
        return "error: user not found"