import json
from typing import Any, Dict

import shopify

from agentorg.env.tools.tools import register_tool
from agentorg.env.tools.shopify.utils import SHOPIFY_AUTH_ERROR

description = "Get the details of collection with a preview of the inventory from a list of collections."
slots = [
    {
        "name": "collection_ids",
        "type": "array",
        "items": {"type": "string"},
        "description": "The collection id, such as 'gid://shopify/Collection/2938501948327'. If there is only 1 collection, return in list with single item. If there are multiple collection ids, please return all of them in a list.",
        "prompt": "In order to proceed, please provide the collection id.",
        "required": True,
    },
    {
        "name": "limit",
        "type": "int",
        "description": "Maximum number of products to show.",
        "prompt": "",
        "required": False
    }
]
outputs = [
    {
        "name": "collection_details",
        "type": "dict",
        "description": "The collection details of the collection. such as \"['{'title': 'Beddings and Pillows', 'description': '', 'productsCount': {'count': 6}, 'products': {'nodes': [{'title': 'Red Bedding', 'description': 'color: red', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Yellow Bedding', 'description': 'color: yellow', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Green Bedding', 'description': 'color: green', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Blue Bedding', 'description': 'color: blue', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Red Pillow', 'description': 'color: red', 'id': 'gid://shopify/Product/7296582090865'}], 'pageInfo': {'hasNextPage': true}}}', '{'title': 'Bedding', 'description': '', 'productsCount': {'count': 4}, 'products': {'nodes': [{'title': 'Red Bedding', 'description': 'color: red', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Yellow Bedding', 'description': 'color: yellow', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Green Bedding', 'description': 'color: green', 'id': 'gid://shopify/Product/7296582090865'}, {'title': 'Blue Bedding', 'description': 'color: blue', 'id': 'gid://shopify/Product/7296582090865'}], 'pageInfo': {'hasNextPage': false}}}']\""
    }
]
COLLECTION_NOT_FOUND = "error: collection not found"
errors = [
    SHOPIFY_AUTH_ERROR,
    COLLECTION_NOT_FOUND
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def preview_collections(collection_ids: list, limit=5, **kwargs) -> str:
    limit = limit or 5
    
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return SHOPIFY_AUTH_ERROR
    
    try:
        ids = ' OR '.join(f'id:{cid.split("/")[-1]}' for cid in collection_ids)
        with shopify.Session.temp(shop_url, api_version, token):
            response = shopify.GraphQL().execute(f"""
                {{
                    collections (query:"{ids}" first: {len(collection_ids)}) {{
                        nodes {{
                            title
                            description
                            productsCount {{
                                count
                            }}
                            products (first: {limit}) {{
                                nodes {{
                                    title
                                    description
                                    id
                                }}
                                pageInfo {{
                                    hasNextPage
                                }}
                            }}
                        }}
                    }}
                }}
            """)
        results = json.loads(response)["data"]["collections"]['nodes']
        return results
    except Exception as e:
        return COLLECTION_NOT_FOUND