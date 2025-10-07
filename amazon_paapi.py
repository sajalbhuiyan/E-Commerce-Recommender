"""Minimal Amazon Product Advertising API v5 helper with AWS SigV4 signing.

Usage:
  Provide credentials (AWS Access Key, Secret Key, and Partner Tag) and call
  search_items(keyword) or get_items(asins).

This file implements only the subset needed by the app and is intentionally
small; be mindful of the PA-API terms and rate limits.
"""
import hashlib
import hmac
import requests
import datetime
import json


def _sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _get_signature_key(key, date_stamp, region_name, service_name):
    k_date = _sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    k_signing = _sign(k_service, 'aws4_request')
    return k_signing


def _make_signed_request(endpoint_host, endpoint_path, region, access_key, secret_key, payload):
    method = 'POST'
    service = 'ProductAdvertisingAPI'
    host = endpoint_host
    endpoint = f'https://{host}{endpoint_path}'

    t = datetime.datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d')

    payload_json = json.dumps(payload)
    canonical_uri = endpoint_path
    canonical_querystring = ''
    content_type = 'application/json; charset=UTF-8'

    canonical_headers = f'content-type:{content_type}\nhost:{host}\nx-amz-date:{amz_date}\n'
    signed_headers = 'content-type;host;x-amz-date'
    payload_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()

    canonical_request = '\n'.join([method, canonical_uri, canonical_querystring, canonical_headers, signed_headers, payload_hash])

    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f'{date_stamp}/{region}/{service}/aws4_request'
    string_to_sign = '\n'.join([algorithm, amz_date, credential_scope, hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()])

    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    authorization_header = f"{algorithm} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

    headers = {
        'Content-Type': content_type,
        'X-Amz-Date': amz_date,
        'Authorization': authorization_header
    }

    resp = requests.post(endpoint, headers=headers, data=payload_json, timeout=15)
    resp.raise_for_status()
    return resp.json()


def search_items(keyword, access_key, secret_key, partner_tag, partner_type='Associates', region='us-east-1', host='webservices.amazon.com', page=1, item_count=10):
    """Search items by keyword (returns list of item dicts)."""
    path = '/paapi5/searchitems'
    resources = [
        'Images.Primary.Medium',
        'ItemInfo.Title',
        'ItemInfo.ByLineInfo',
        'Offers.Listings.Price'
    ]
    payload = {
        'Keywords': keyword,
        'PartnerTag': partner_tag,
        'PartnerType': partner_type,
        'SearchIndex': 'All',
        'Resources': resources,
        'ItemPage': page
    }
    data = _make_signed_request(host, path, region, access_key, secret_key, payload)
    items = data.get('SearchResult', {}).get('Items', [])
    return items


def get_items(asins, access_key, secret_key, partner_tag, partner_type='Associates', region='us-east-1', host='webservices.amazon.com'):
    """GetItems by ASIN list; returns list of item dicts."""
    path = '/paapi5/getitems'
    resources = [
        'Images.Primary.Medium',
        'ItemInfo.Title',
        'ItemInfo.ByLineInfo',
        'Offers.Listings.Price'
    ]
    payload = {
        'ItemIds': asins,
        'PartnerTag': partner_tag,
        'PartnerType': partner_type,
        'Resources': resources
    }
    data = _make_signed_request(host, path, region, access_key, secret_key, payload)
    items = data.get('ItemsResult', {}).get('Items', [])
    return items
