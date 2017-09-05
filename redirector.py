"""

EmailRedirector download unseen email from "From" mailbox.
Emails can be processed and then uploaded to "To" mailbox.
If attachment type is not 'image' -> upload to OwnCloud server and return link.

KNOWN ISSUES:
    1. Wrong encoding for mailbox folders (possibly utf-7), which not supported by imaplib

"""

import os
import time
import email
import pathlib
import imaplib
import requests

from requests.exceptions import ConnectionError

import logging
import traceback
import mimetypes

from urllib import parse
from xml.etree import ElementTree
from uuid import uuid4 as random_uuid

from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from email.header import decode_header
from email.encoders import encode_base64

from settings import LOGGING_DIR, LOG_LEVEL, LOG_FORMATTER


logger = logging.getLogger(__name__)
handler = logging.FileHandler(LOGGING_DIR)
logger.setLevel(LOG_LEVEL)
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter(LOG_FORMATTER)
handler.setFormatter(formatter)
logger.addHandler(handler)


class OwnCloudPusher:

    """
    Upload files to OwnCloud

    """

    def __init__(self, login, password):

        """
        Initialize incoming data

        :param login: account name
        :type login: str

        :param password: account password
        :type password: str

        """

        self._base_url = 'https://owncloud.domain.com/'
        self._dav_url = self._base_url + 'remote.php/dav/files/' + login
        self._api_url = self._base_url + 'ocs/v1.php/'
        self._attempts = 0

        self.session = requests.session()
        self.session.auth = (login, password)

        self.logger = logger

    def push_file(self, file_path, remote_path):

        """
        Pushing file to OwnCloud server and return link to file

        :param file_path: (required) path to local file
        :type file_path: str

        :param remote_path: (required) remote path
        :type remote_path: str

        :return: link to uploaded file or False
        :rtype: str or bool

        """

        dir_created = self._make_dir(remote_path)

        if not dir_created:
            return False

        remote_path += os.path.basename(file_path)

        file = pathlib.Path(file_path)

        if not file.is_file():
            self.logger.error('ERROR: File not found: {0}'.format(file_path))
            return False

        if file.stat().st_size <= 0:
            self.logger.error('ERROR: File is empty: {0}'.format(file_path))
            return False

        try:
            with file.open(mode='rb') as file_handle:
                uploaded = self._dav_request('PUT',
                                             remote_path,
                                             data=file_handle)

        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

        else:
            if uploaded:
                self.logger.debug('File uploaded: {0}'.format(os.path.basename(file_path)))

                xml_text = self._api_request(
                    'POST',
                    'apps/files_sharing/api/v1',
                    'shares',
                    data={'shareType': 3, 'path': remote_path}
                )

                if xml_text:
                    root = ElementTree.fromstring(xml_text)

                    for node in root.iter('url'):
                        self.logger.debug('Link: {0}'.format(node.text))
                        return node.text  # link to file
                else:
                    self.logger.error('ERROR: Link failure: {0}'.format(file_path))
                    return False
            else:
                self.logger.error('ERROR: File upload failure: {0}'.format(file_path))
                return False

        self.logger.error('ERROR: Unexpected error while uploading')
        return False

    def _make_dir(self, path):

        """
        Creates remote directory on specified path

        :param path: (required) path to directory
        :type path: str

        :return: True if success, False otherwise
        :rtype: bool

        """

        created = self._dav_request('MKCOL', path)

        if created:
            self.logger.debug('Successfully created remote directory: {0}'.format(path))
            return True

        self.logger.error('ERROR: Unable to create remote directory: {0}'.format(path))
        return False

    def _dav_request(self, method, path, **kwargs):

        """
        Send WebDAV request: creates directory or upload file

        :param method: (required) HTTP method, eg. 'PUT'
        :type method: str

        :param path: (required) path to remote location (file or directory)
        :type path: str

        :param kwargs: (optional) any params that requests.Request accepts

        :return: True if success, False otherwise
        :rtype: bool

        """

        try:
            res = self.session.request(method,
                                       self._dav_url + parse.quote(path.encode('utf-8')),
                                       **kwargs)

        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

        else:
            if res.status_code in [201, 204, 405]:
                return True
            return False
        return False

    def _api_request(self, method, service, action, **kwargs):

        """
        Send OwnCloud API request: share file with link

        :param method: (required) HTTP method, eg. 'PUT'
        :type method: str

        :param service: path to OwnCloud service, eg. 'apps/files_sharing/api/v1'
        :type service: str

        :param action: which action perform, eg. 'shares'
        :type service: str

        :param data: (optional) share type and path to place file
                     eg. { 'shareType': {int} 3, -- means 'share with link'
                           'path': {str} 'path/to/remote/file' }
        :type data: dict

        :param kwargs: (optional) any params that requests.Request accepts

        :return: link to file on success, False otherwise
        :rtype: bool or str

        """

        path = self._api_url + service + '/' + action
        attributes = kwargs.copy()

        if 'headers' not in attributes:
            attributes['headers'] = {}

        attributes['headers']['OCS-APIREQUEST'] = 'true'

        try:
            res = self.session.request(method, path.strip(), **attributes)

        except ConnectionError as error:
            if self._attempts <= 5:
                self.logger.error('WARNING: attempt #{1} of link obtaining failed: {0}'.format(error, self._attempts))
                self._attempts += 1
                return self._api_request(method, service, action, **kwargs)
            else:
                self.logger.error('ERROR: all {1} attempts failed, last error: {0}'.format(error, self._attempts))
                raise error

        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

        else:
            if res.status_code in [200]:
                self._attempts = 0
                return res.text
            else:
                self.logger.error('ERROR: wrong OwnCloud response: {0}, {1}'.format(res.status_code, res.text))
                return False

        return False


class EmailRedirector:

    """
    Read messages from 'From' mailbox & redirect to 'To' mailbox

    """

    def __init__(self, from_box, to_box, place_to, owncloud, project_title, skip_topics=None):

        """
        Initialize incoming data

        :param from_box: (required) auth data for 'From' mailbox
        :type from_box: dict, eg. { 'server': 'imap_server_url',
                                    'login': 'user_login',
                                    'password': 'user_password' }

        :param to_box: (required) auth data for 'To' mailbox
        :type to_box: dict, eg. { 'server': 'imap_server_url',
                                  'login': 'user_login',
                                  'password': 'user_password' }

        :param place_to: (required) where temporary save attachments
        :type place_to: str, eg. 'path/to/temporary/folder'

        :param owncloud: (required) auth data for OwnCloud server
        :type owncloud: dict, eg. { 'login': 'user_login',
                                    'password': 'user_password' }

        :param project_title: (required) project title, uses as OwnCloud folder name
        :type project_title: str, eg. 'my_awesome_project'

        :param skip_topics: (optional): email topics to skip
        :type skip_topics: tuple

        """

        self._from_box = from_box
        self._to_box = to_box
        self._place_to = place_to
        self._owncloud = OwnCloudPusher(**owncloud)
        self._project = project_title
        self._skip_topics = skip_topics

        self._html_template = '''<!DOCTYPE html>
                                 <html>
                                    <head>
                                        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                                        <style>
                                            #redirector-block {{
                                                border: 1px dashed black;
                                                background-color: #f2f2f2;
                                                padding: 5px;
                                            }}                                                
                                        </style>
                                    </head>
                                    <body>
                                        {content}
                                    </body>
                                 </html>'''

        self._logged_in = False
        self.logger = logger

    def __enter__(self):

        """
        Call, after object initialize

        :return: EmailRedirector instance
        :rtype: EmailRedirector

        """

        self._open_connection()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        """
        Call, before exiting object

        :param exc_type: incoming Exception type
        :param exc_val: incoming Exception value
        :param exc_tb: incoming Exception traceback

        :return: False if exception occurred
        :raise: incoming Exception

        """

        self._close_connection()

        return False  # re-raise exception if occurred

    def _open_connection(self):

        """
        Connect to 'From' & 'To' mailboxes

        """

        if not self._logged_in:
            try:
                self.from_server = imaplib.IMAP4_SSL(self._from_box['server'])
                self.from_server.login(self._from_box['login'], self._from_box['password'])

                self.to_server = imaplib.IMAP4_SSL(self._to_box['server'])
                self.to_server.login(self._to_box['login'], self._to_box['password'])

            except Exception:
                formatted_lines = traceback.format_exc().splitlines()
                self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

            else:
                self.logger.debug('Logged in to "From" mailbox: {0}'.format(self._from_box['login']))
                self.logger.debug('Logged in to "To" mailbox: {0}'.format(self._to_box['login']))
                self._logged_in = True

    def _close_connection(self):

        """
        Disconnect from 'From' & 'To' mailboxes

        """

        if self._logged_in:
            try:
                self.from_server.logout()
                self.to_server.logout()

            except Exception:
                formatted_lines = traceback.format_exc().splitlines()
                self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

            else:
                self.logger.debug('Logged out from "From" mailbox: {0}'.format(self._from_box['login']))
                self.logger.debug('Logged out from "To" mailbox: {0}'.format(self._to_box['login']))

            finally:
                self._logged_in = False

    def fetch_unread_messages(self):

        """
        Retrieve unread messages from 'INBOX' & all subdirectories

        :returns: simple messages representation
        :rtype: generator

        """

        try:
            for directory in self._mailbox_dirs(self.from_server):
                self.from_server.select(directory, readonly=False)
                get_messages, messages = self.from_server.search(None, '(UNSEEN)')

                if get_messages == 'OK':
                    if messages[0] != b'':
                        self.logger.debug('{0}: got new message'.format(directory))

                        for msg_number in messages[0].split():
                            result, raw_msg = self.from_server.fetch(msg_number, '(RFC822)')

                            if result == 'OK':
                                msg_object = email.message_from_bytes(raw_msg[0][1])
                                subject = self._decode(msg_object['Subject'])
                                skip_this = False

                                self.logger.debug('{0}: message subject: {1}'.format(directory, subject))

                                if self._skip_topics is not None:
                                    for topic in self._skip_topics:
                                        if subject.startswith(topic):
                                            self.logger.debug('{0}: message skipped with topic: {1}'.format(directory,
                                                                                                            subject))
                                            self.from_server.store(msg_number, '+FLAGS', '\\SEEN')  # Make message 'seen'

                                            skip_this = True
                                            break

                                if not skip_this:
                                    yield self._extractor(msg_object, msg_number)

                            else:
                                self.logger.error('ERROR: {0}: return BAD status - {1}, {2}'.format(directory,
                                                                                                    result,
                                                                                                    raw_msg))
                    else:
                        self.logger.debug('{0}: no new messages'.format(directory))
                else:
                    self.logger.error('ERROR: {0}: return BAD status - {1}, {2}'.format(directory,
                                                                                        get_messages,
                                                                                        messages))

        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))
            self._close_connection()

    def _mailbox_dirs(self, mailbox):

        """
        Retrieve ALL folders in 'INBOX' from connection handler

        :param mailbox: (required) IMAP4 connection handler
        :type mailbox: IMAP4 client instance

        :returns: directories names
        :rtype: str

        """

        yield '"INBOX"'  # root directory

        try:
            list_dirs = mailbox.list('INBOX')

        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))
            self._close_connection()

        else:
            for dir_name in list_dirs[1]:
                yield '"' + dir_name.split(b'"')[-2].decode('utf-8') + '"'  # '"INBOX/directory (support spaces)"'

    def _extractor(self, msg_object, msg_number):

        """
        Extract parts from email object

        :param msg_object: (required) email object
        :type msg_object: email.message.Message instance

        :return: simple message representation
                 eg. { 'uuid': {str} 'message_uuid',
                       'links': {list of tuples} [('file_name', 'owncloud_link'), ...],
                       'parts': {list of objects} [email.message.MIMEBase, ...],
                       'attach': {list of tuples} [('path/to/file', 'content_type'), ...],
                       'object': {object} email.message.Message }
        :rtype: dict

        """

        # message ID
        uuid = random_uuid().hex

        links = list()
        parts = list()
        attach = list()

        attachments_counter = 1

        # iterate over ALL message parts (include nested)
        for part in self._extract_parts(msg_object):
            content_type = part.get_content_type()
            base, spec = content_type.split('/', 1)

            # text or html = save 'as is'
            if base == 'text':
                charset = part.get_content_charset()

                if charset is not None:
                    payload = (part.get_payload(decode=True)).decode(charset)

                    if spec == 'plain':
                        new_part = MIMEText(payload, 'plain', 'utf-8')
                        parts.append(new_part)

                    if spec == 'html':
                        new_part = MIMEText(payload, 'html', 'utf-8')
                        parts.append(new_part)

                continue

            # save attach to temporary folder
            attach_name = self._decode(part.get_filename())
            attach_path = os.path.join(self._place_to, uuid + '_' + str(attachments_counter) +
                                       '.' + attach_name.split('.')[-1])

            if not os.path.isdir(attach_path):
                payload = part.get_payload(decode=True)

                try:
                    if payload is not None:
                        with open(attach_path, mode='wb') as file_pointer:
                            file_pointer.write(payload)

                except Exception:
                    formatted_lines = traceback.format_exc().splitlines()
                    self.logger.error('EXCEPTION: \n{exc} PART: {part}'.format(exc='\n'.join(formatted_lines),
                                                                               part=payload))

            # increment attachments counter
            attachments_counter += 1

            # image = append to compression list
            if base == 'image':
                attach.append((attach_path, content_type))
                continue

            # not image = upload to OwnCloud, return link to file
            else:
                link = self._owncloud.push_file(attach_path, '/' + self._project + '/')

                if link:
                    links.append((attach_name, link))

                    try:
                        os.remove(attach_path)
                    except Exception:
                        formatted_lines = traceback.format_exc().splitlines()
                        self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

                continue

        return {
            'uuid': uuid,
            'links': links,
            'parts': parts,
            'attach': attach,
            'object': msg_object,
            'msg_number': msg_number
        }

    def _extract_parts(self, msg_object):

        """
        Recursively extract parts from original message object

        :param msg_object: (required) email object
        :type msg_object: email.message.Message instance

        :return: message part object
        :rtype: email.message.Message instance

        """

        for part in msg_object.walk():
            if part.get_content_maintype() == 'multipart':
                self._extract_parts(part)
            else:
                yield part

    def redirect(self, mail, new_paths=list(), broken=list()):

        """
        Redirect processed email to "To" mailbox & remove temporary files

        :param mail: (required) simple representation of email object
        :type mail: dict

        :param new_paths: (optional) new paths to converted files
        :type new_paths: list

        """

        try:
            msg = self._build_message(mail, new_paths, broken)

            self.to_server.select(readonly=False)
            self.to_server.append('INBOX', '', time.time(), bytes(msg))

        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

        else:
            self.logger.debug('Message {0} with subject '
                              '"{1}" successfully redirected'.format(mail['uuid'],
                                                                     self._decode(mail['object']['Subject'])))
            self.from_server.store(mail['msg_number'], '+FLAGS', '\\SEEN')  # Make message 'seen'

            for i in mail['attach']:
                try:
                    os.remove(i[0])
                except Exception:
                    formatted_lines = traceback.format_exc().splitlines()
                    self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

    def _decode(self, to_decode):

        """
        Decode input header to extracted encoding

        :param to_decode: (required) header to decode
        :type to_decode: bytes

        :return: Decoded text
        :rtype: str

        """

        if to_decode is not None:
            text = decode_header(to_decode)[0][0]
            encoding = decode_header(to_decode)[0][1]

            if isinstance(text, bytes):
                if encoding is not None:
                    return text.decode(encoding)
                else:
                    return text.decode('utf-8')
            else:
                return text
        else:
            self.logger.error('ERROR: to_decode: \n{0}'.format(to_decode))
            return "none.type"

    def _build_message(self, mail, new_paths, broken):

        """
        Format new message object

        :param mail: (required) simple representation of email object
        :type mail: dict

        :param new_paths: (required) paths to compressed images
        :type new_paths: list

        :param broken: (required) paths to broken files
        :type broken: list

        :return: email object: original's message 'From', 'To', 'Subject' + OwnCloud links Block +
                 + original's message body (both plain & html) + Message ID Block + processed attachments
        :rtype: email.message.Message instance

        """

        msg = MIMEMultipart()

        msg['To'] = mail['object']['To']
        msg['From'] = mail['object']['From']
        msg['Subject'] = mail['object']['Subject']

        # owncloud links
        if mail['links'] or broken:
            links_format = '<li><a href="{link}">{file_name}</a> [{link}]</li>'

            attach_gen = (links_format.format(file_name=f_name, link=link) for f_name, link in mail['links'])

            brokens = list()

            if broken:
                for file in broken:
                    link = self._owncloud.push_file(file, '/' + self._project + '/')

                    if link:
                        brokens.append((os.path.basename(file), link))
                        try:
                            os.remove(file)
                        except Exception:
                            formatted_lines = traceback.format_exc().splitlines()
                            self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

            broke_gen = (links_format.format(file_name=f_name, link=link) for f_name, link in brokens)

            links = '''<div id="redirector-block">
                           {attaches}
                           {brokens}                           
                       </div>'''.format(
                                    attaches="<b>Вложения:</b>"
                                             "<ol>{links_list}</ol>".format(links_list=''.join(attach_gen)),
                                    brokens="<b>Поврежденные:</b>"
                                            "<ol>{links_list}</ol>".format(links_list=''.join(broke_gen))
                                            if broken else "")

            msg.attach(MIMEText(self._html_template.format(content=links), 'html', 'utf-8'))

        # original's message parts
        if mail['parts']:
            for part in mail['parts']:
                msg.attach(part)

        # info block
        info_block = '<div id="redirector-block"><b>ID:</b> {id}</div>'.format(id=mail['uuid'])
        msg.attach(MIMEText(self._html_template.format(content=info_block), 'html', 'utf-8'))

        # replace attachments paths
        if new_paths:
            mail['attach'] = [x for x in self._replace_paths(new_paths)]

        # attachments
        if mail['attach']:
            for path, content_type in mail['attach']:
                try:
                    assert isinstance(content_type, str), "content_type: type mismatch"

                except AssertionError as err:
                    self.logger.error('ASSERT: {2} content_type: {0}: {1}'.format(type(content_type), content_type, err))

                else:
                    base, spec = content_type.split('/', 1)
                    part = MIMEBase(base, spec)

                    try:
                        with open(path, mode='rb') as file_pointer:
                            part.set_payload(file_pointer.read())

                    except Exception:
                        formatted_lines = traceback.format_exc().splitlines()
                        self.logger.error('EXCEPTION: \n{exc}'.format(exc='\n'.join(formatted_lines)))

                    encode_base64(part)
                    part.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(path)))

                    msg.attach(part)

        return msg

    @staticmethod
    def _replace_paths(new):

        """
        Replace paths + add content-type to path

        :param new: list of new paths
        :type new: list

        :return: tuple of path & content-type, eg. ('path/to/image.jpg', 'image/jpeg')
        :rtype: generator

        """

        for path in new:
            yield path, mimetypes.guess_type(path)[0]


'''
while True:
    with EmailRedirector(
            {
                'server': 'mail.domain.ru',
                'login': 'user1',
                'password': 'pass1'
            },
            {
                'server': 'mail.domain.ru',
                'login': 'user2',
                'password': 'pass2'
            },
            'path/to/files',
            {
                'login': 'login',
                'password': 'password'
            },
            'project') as red:

        for x in red.fetch_unread_messages():
            # some processing stuff ...
            red.redirect(x)

    time.sleep(1)
'''
