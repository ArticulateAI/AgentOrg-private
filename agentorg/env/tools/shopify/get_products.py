import json
from typing import Any, Dict

import shopify

from agentorg.env.tools.tools import register_tool
from agentorg.env.tools.shopify.utils import SHOPIFY_AUTH_ERROR

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
        "description": "The product details of each products. such as \"[{'id': 'gid://shopify/Product/7296581894257', 'title': 'Nordic Bedding Set', 'description': 'size: 48 cm', 'totalInventory': 3, 'category': 'bedding', 'variants': {'nodes': [{'price': '50.99'}]}}, {'id': 'gid://shopify/Product/7296582123633', 'title': 'Ocean Theme Bedding ', 'description': 'Grade A', 'totalInventory': 0, 'category': 'bedding', 'variants': {'nodes': [{'price': '76.99'}]}}]\".",
    }
]
PRODUCTS_NOT_FOUND = "error: product not found"
errors = [
    SHOPIFY_AUTH_ERROR,
    PRODUCTS_NOT_FOUND
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_products(product_ids: list, **kwargs) -> str:
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return SHOPIFY_AUTH_ERROR
    
    try:
        with shopify.Session.temp(shop_url, api_version, token):
            ids = ' OR '.join(f'id:{pid.split("/")[-1]}' for pid in product_ids)
            response = shopify.GraphQL().execute(f"""
                {{
                    products (query:"{ids}" first: {len(product_ids)}) {{
                        nodes {{
                            id
                            title
                            description
                            totalInventory
                            category {{
                                name
                            }}
                            variants (first: 1) {{
                                nodes {{
                                    compareAtPrice
                                }}
                            }}
                        }}
                    }}
                }}
            """)
        response = json.loads(response)["data"]["products"]["nodes"]
        response_text = "After searching, products info is as below:\n"
        # TODO: temp change for better response
        for product in response:
            response_text += f"Product Title: {product.get('title')}\n"
            response_text += f"Product Description: {product.get('description')}\n"
            response_text += f"Total Inventory: {product.get('totalInventory')}\n"
        return response_text
    except Exception as e:
        return PRODUCTS_NOT_FOUND
    
