import json
from typing import Any, Dict
import shopify
import logging

from agentorg.env.tools.shopify.utils import *
from agentorg.env.tools.tools import register_tool
from agentorg.env.tools.shopify.utils import SHOPIFY_AUTH_ERROR

logger = logging.getLogger(__name__)

description = "Search collections by string query. If no collections are found, the function will return an error message."
slots = [
    {
        "name": "query",
        "type": "string",
        "description": "The string query to search collections, such as 'Hats'. If query is empty string, it returns all collections.",
        "prompt": "In order to proceed, please provide a query for the collections search.",
        "required": False,
    },
] + PAGEINFO_SLOTS
outputs = [
    {
        "name": "collections_list",
        "type": "dict",
        "description": "A list of up to limit number of collections that satisfies the query. Such as \"[{'id': 'gid://shopify/Collection/7296580845681'}, {'id': 'gid://shopify/Collection/7296580878449'}, {'id': 'gid://shopify/Collection/7296581042289'}]\"",
    }
] + PAGEINFO_OUTPUTS
COLLECTION_SEARCH_ERROR = "error: collection search failed"
NO_COLLECTIONS_FOUND_ERROR = "no collections found"

errors = [
    SHOPIFY_AUTH_ERROR,
    COLLECTION_SEARCH_ERROR,
    NO_COLLECTIONS_FOUND_ERROR,
    NAVIGATE_WITH_NO_CURSOR,
    NO_NEXT_PAGE,
    NO_PREV_PAGE
]

@register_tool(description, slots, outputs, lambda x: x[0] not in errors)
def search_collections(query: str, limit=10, navigate='stay', pageInfo=None, **kwargs) -> str:
    limit = limit or 10
    navigate = navigate or 'stay'
    
    shop_url = kwargs.get("shop_url")
    api_version = kwargs.get("api_version")
    token = kwargs.get("token")

    if not shop_url or not api_version or not token:
        return SHOPIFY_AUTH_ERROR, ''
    
    logger.info(f"PARAMS: {query, limit, navigate, pageInfo}")
    
    nav = f'first: {limit}'
    if navigate and navigate != 'stay':
        if not pageInfo:
            return NAVIGATE_WITH_NO_CURSOR, ''
        
        if navigate == 'next':
            if not pageInfo['hasNextPage']:
                return NO_NEXT_PAGE, ''
            nav = f"first: {limit}, after: \"{pageInfo['endCursor']}\""
            
        elif navigate == 'prev': 
            if not pageInfo['hasPreviousPage']:
                return NO_PREV_PAGE, ''
            nav = f"last: {limit}, before: \"{pageInfo['startCursor']}\""
    
    try:
        with shopify.Session.temp(shop_url, api_version, token):
            response = shopify.GraphQL().execute(f"""
                {{
                    collections ({nav}, query: "{query}") {{
                        nodes {{
                            id
                        }}
                        pageInfo {{
                            endCursor
                            hasNextPage
                            hasPreviousPage
                            startCursor
                        }}
                    }}
                }}
        """)
        
        data = json.loads(response)['data']['collections']
        nodes = data['nodes']
        pageInfo = data['pageInfo']
        if len(nodes):
            return nodes, pageInfo
        else:
            return NO_COLLECTIONS_FOUND_ERROR, ''
    
    except Exception as e:
        return COLLECTION_SEARCH_ERROR, ''