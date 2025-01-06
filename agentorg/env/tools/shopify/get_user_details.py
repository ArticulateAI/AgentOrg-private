import json
from typing import Any, Dict
import shopify

from agentorg.env.tools.tools import register_tool

description = "Get the details of a user."
slots = [
    {
        "name": "user_id",
        "type": "string",
        "description": "The user id, such as 'gid://shopify/Customer/13573257450893'.",
        "prompt": "In order to proceed, Could you please provide the user id?",
        "required": True,
    }
]
outputs = [
    {
        "name": "user_details",
        "type": "dict",
        "description": "The user details of the user. such as '{\"firstName\": \"John\", \"lastName\": \"Doe\", \"email\": \"example@gmail.com\"}'."
    }
]

@register_tool(description, slots, outputs)
def get_user_details(user_id: str, **kwargs) -> str:
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return "error: missing some or all required shopify authentication parameters: shop_url, api_version, token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"

    try:
        with shopify.Session.temp(shop_url, api_version, token):
            response = shopify.GraphQL().execute(f"""
            {{
                customer(id: "{user_id}") {{
                    firstName
                    lastName
                    email
                    phone
                    numberOfOrders
                    amountSpent {{
                        amount
                        currencyCode
                    }}
                    createdAt
                    updatedAt
                    note
                    verifiedEmail
                    validEmailAddress
                    tags
                    lifetimeDuration
                    defaultAddress {{
                        formattedArea
                        address1
                    }}
                    addresses {{
                        address1
                    }}
                    orders (first: 10) {{
                        edges {{
                            node {{
                                id         
                            }}
                        }}
                    }}
                }}
            }}
            """)
        parsed_response = json.loads(response)["data"]["customer"]
        return json.dumps(parsed_response)
    except Exception as e:
        print("error: user not found")
        print(e)
        return "error: user not found"
