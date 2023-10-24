import base64
import datetime
import ipaddress
import json

import chardet
import pyarrow as pa

from .parser import Parser


class CensysParser(Parser):
    """Class for parsing examples from the Censys Universal Internet Dataset.

    Parses examples (lines) from a Censys Universal Internet Dataset snapshot
    exported from BigQuery.

    Attributes:
        schema: Arrow schema corresponding to parsed examples.
        sort_column: Column name to use for sorting examples.
    """

    def __init__(self,
                 detect_encoding: bool = False,
                 default_encoding: str = 'utf-8'):
        """Creates a parser.

        Args:
            detect_encoding: Whether to try and detect the encoding of banners
                for converting them to unicode strings. If True,
                `default_encoding` is used when encoding detection fails,
                replacing errors with a unicode replacement character (U+FFFD).
                If False, each byte is converted to the corresponding unicode
                character, e.g., 0xF0 -> U+00F0.
            default_encoding: Encoding to use to when encoding detection fails.
        """
        self.detect_encoding = detect_encoding
        self.default_encoding = 'utf-8'

    def _get_schema(self) -> pa.Schema:
        # Returns the Arrow schema corresponding to parsed examples.
        return pa.schema({
            'id':
                pa.binary(),
            'ip':
                pa.string(),
            'port':
                pa.uint16(),
            'service_name':
                pa.string(),
            'banner':
                pa.string(),
            'software':
                pa.list_(
                    pa.struct({
                        'part':
                            pa.string(),
                        'vendor':
                            pa.string(),
                        'product':
                            pa.string(),
                        'version':
                            pa.string(),
                        'other':
                            pa.list_(
                                pa.struct({
                                    'key': pa.string(),
                                    'value': pa.string()
                                }))
                    })),
            'truncated':
                pa.bool_(),
            'observed_at':
                pa.timestamp('ms', tz='UTC')
        })

    def _get_sort_column(self) -> str:
        # Returns the column name to use for sorting parsed examples.
        return 'id'

    def parse(self, line: str) -> dict:
        """Parses and returns an example (i.e., a single line from a data file).

        Args
            line: Line to parse.

        Returns:
            Parsed example.
        """
        example = json.loads(line)
        ip = example['host_identifier']
        if 'ipv4' in ip:
            ip = ip['ipv4']
            id_ = b'\x00' + ipaddress.IPv4Address(ip).packed
        else:
            ip = ip['ipv6']
            id_ = b'\x01' + ipaddress.IPv6Address(ip).packed

        port = int(example['port'])
        id_ += port.to_bytes(2, 'big')

        banner = base64.b64decode(example.get('banner', ''))
        if self.detect_encoding:
            try:
                banner = banner.decode('ascii')
            except:
                encoding = chardet.detect(banner)['encoding']
                encoding = encoding or self.default_encoding
                try:
                    banner = banner.decode(encoding, errors='replace')
                except LookupError:
                    banner = banner.decode(self.default_encoding,
                                           errors='replace')
        else:
            banner = ''.join([chr(byte) for byte in banner])

        software = []
        for s in example['software']:
            other = [{
                'key': o['key'],
                'value': o['value']
            } for o in s.get('other', [])]
            software.append({
                'part': s.get('part', ''),
                'vendor': s.get('vendor', ''),
                'product': s.get('product', ''),
                'version': s.get('version', ''),
                'other': other
            })

        try:
            observed_at = datetime.datetime.strptime(
                example['observed_at'], '%Y-%m-%d %H:%M:%S.%f UTC')
        except ValueError:
            observed_at = datetime.datetime.strptime(example['observed_at'],
                                                     '%Y-%m-%d %H:%M:%S UTC')

        return {
            'id': id_,
            'ip': ip,
            'port': port,
            'service_name': example['service_name'],
            'banner': banner,
            'software': software,
            'truncated': example['truncated'],
            'observed_at': observed_at
        }
