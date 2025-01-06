import json
from typing import Any, Dict

import shopify

from agentorg.env.tools.tools import register_tool

description = "Get the status and details of an order."
slots = [
    {
        "name": "order_ids",
        "type": "array",
        "items": {"type": "string"},
        "description": "The order id, such as gid://shopify/Order/1289503851427. If there is only 1 order, return in list with single item. If there are multiple order ids, please return all of them in a list.",
        "prompt": "Please provide the order id to get the details of the order.",
        "required": True,
    }
]
outputs = [
    {
        "name": "order_details",
        "type": "dict",
        "description": "The order details of the order. such as '{\"id\": \"gid://shopify/Order/1289503851427\", \"name\": \"#1001\", \"totalPriceSet\": {\"presentmentMoney\": {\"amount\": \"10.00\"}}, \"lineItems\": {\"edges\": [{\"node\": {\"id\": \"gid://shopify/LineItem/1289503851427\", \"title\": \"Product 1\", \"quantity\": 1, \"variant\": {\"id\": \"gid://shopify/ProductVariant/1289503851427\", \"product\": {\"id\": \"gid:////shopify/Product/1289503851427\"}}}}]}}'.",
    }
]

@register_tool(description, slots, outputs)
def get_order_details(order_ids: list, **kwargs) -> str:
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return "error: missing some or all required shopify authentication parameters: shop_url, api_version, token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"
    
    try:
        with shopify.Session.temp(shop_url, api_version, token):
            results = []
            for order_id in order_ids:
                response = shopify.GraphQL().execute(f"""
                {{
                    order (id: "{order_id}") {{
                        id
                        name
                        totalPriceSet {{
                            presentmentMoney {{
                                amount
                            }}
                        }}
                        lineItems(first: 10) {{
                            edges {{
                                node {{
                                    id
                                    title
                                    quantity
                                    variant {{
                                        id
                                        product {{
                                            id
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                """)
            parsed_response = json.loads(response)["data"]["order"]
            results.append(json.dumps(parsed_response))
        return results
    except Exception as e:
        return "error: order not found"
