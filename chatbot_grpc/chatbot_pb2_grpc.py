# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import chatbot_grpc.chatbot_pb2 as chatbot__pb2


class ChatBotServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Chatbot = channel.unary_unary(
            '/ChatBotService/Chatbot',
            request_serializer=chatbot__pb2.ReceivedMessage.SerializeToString,
            response_deserializer=chatbot__pb2.ResponsedMessage.FromString,
        )


class ChatBotServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Chatbot(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ChatBotServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Chatbot': grpc.unary_unary_rpc_method_handler(
            servicer.Chatbot,
            request_deserializer=chatbot__pb2.ReceivedMessage.FromString,
            response_serializer=chatbot__pb2.ResponsedMessage.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'ChatBotService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class ChatBotService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Chatbot(request,
                target,
                options=(),
                channel_credentials=None,
                call_credentials=None,
                insecure=False,
                compression=None,
                wait_for_ready=None,
                timeout=None,
                metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ChatBotService/Chatbot',
                                             chatbot__pb2.ReceivedMessage.SerializeToString,
                                             chatbot__pb2.ResponsedMessage.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
