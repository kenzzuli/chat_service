# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: chatbot.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='chatbot.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rchatbot.proto\"M\n\x0fReceivedMessage\x12\x0f\n\x07user_id\x18\x01 \x01(\t\x12\x14\n\x0cuser_message\x18\x02 \x01(\t\x12\x13\n\x0b\x63reate_time\x18\x03 \x01(\x05\">\n\x10ResponsedMessage\x12\x15\n\ruser_response\x18\x01 \x01(\t\x12\x13\n\x0b\x63reate_time\x18\x02 \x01(\x05\x32@\n\x0e\x43hatBotService\x12.\n\x07\x43hatbot\x12\x10.ReceivedMessage\x1a\x11.ResponsedMessageb\x06proto3'
)




_RECEIVEDMESSAGE = _descriptor.Descriptor(
  name='ReceivedMessage',
  full_name='ReceivedMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='user_id', full_name='ReceivedMessage.user_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user_message', full_name='ReceivedMessage.user_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='create_time', full_name='ReceivedMessage.create_time', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=94,
)


_RESPONSEDMESSAGE = _descriptor.Descriptor(
  name='ResponsedMessage',
  full_name='ResponsedMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='user_response', full_name='ResponsedMessage.user_response', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='create_time', full_name='ResponsedMessage.create_time', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=96,
  serialized_end=158,
)

DESCRIPTOR.message_types_by_name['ReceivedMessage'] = _RECEIVEDMESSAGE
DESCRIPTOR.message_types_by_name['ResponsedMessage'] = _RESPONSEDMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReceivedMessage = _reflection.GeneratedProtocolMessageType('ReceivedMessage', (_message.Message,), {
  'DESCRIPTOR' : _RECEIVEDMESSAGE,
  '__module__' : 'chatbot_pb2'
  # @@protoc_insertion_point(class_scope:ReceivedMessage)
  })
_sym_db.RegisterMessage(ReceivedMessage)

ResponsedMessage = _reflection.GeneratedProtocolMessageType('ResponsedMessage', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSEDMESSAGE,
  '__module__' : 'chatbot_pb2'
  # @@protoc_insertion_point(class_scope:ResponsedMessage)
  })
_sym_db.RegisterMessage(ResponsedMessage)



_CHATBOTSERVICE = _descriptor.ServiceDescriptor(
  name='ChatBotService',
  full_name='ChatBotService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=160,
  serialized_end=224,
  methods=[
  _descriptor.MethodDescriptor(
    name='Chatbot',
    full_name='ChatBotService.Chatbot',
    index=0,
    containing_service=None,
    input_type=_RECEIVEDMESSAGE,
    output_type=_RESPONSEDMESSAGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_CHATBOTSERVICE)

DESCRIPTOR.services_by_name['ChatBotService'] = _CHATBOTSERVICE

# @@protoc_insertion_point(module_scope)
