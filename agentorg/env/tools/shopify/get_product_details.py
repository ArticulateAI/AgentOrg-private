import json
from typing import Any, Dict

import shopify

from agentorg.env.tools.tools import register_tool

description = "Get the inventory information and description details of a product."
slots = [
    {
        "name": "product_ids",
        "type": "array",
        "items": {"type": "string"},
        "description": "The product id, such as 'gid://shopify/Product/2938501948327'. If there is only 1 product, return in list with single item. If there are multiple product ids, please return all of them in a list.",
        "prompt": "In order to proceed, please provide the product id.",
        "required": True,
    }
]
outputs = [
    {
        "name": "product_details",
        "type": "dict",
        "description": "The product details of the product. such as '{\"id\": \"gid://shopify/Product/2938501948327\", \"title\": \"Product 1\", \"description\": \"This is a product description.\"}'.",
    }
]

@register_tool(description, slots, outputs)
def get_product_details(product_ids: list, **kwargs) -> str:
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return "error: missing some or all required shopify authentication parameters: shop_url, api_version, token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"
    
    try:
        with shopify.Session.temp(shop_url, api_version, token):
            results = []
            for product_id in product_ids:
                response = shopify.GraphQL().execute(f"""
                {{
                    product (id: "{product_id}") {{
                        id
                        title
                        description
                        totalInventory
                    }}
                }}
                """)
            parsed_response = json.loads(response)["data"]["product"]
            results.append(json.dumps(parsed_response))
        return results
    except Exception as e:
        return "error: product not found"
