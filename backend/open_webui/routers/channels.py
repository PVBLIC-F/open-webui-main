import json
import logging
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from pydantic import BaseModel

from open_webui.socket.main import (
    emit_to_users,
    enter_room_for_users,
    sio,
    get_user_ids_from_room,
)
from open_webui.models.users import (
    UserIdNameResponse,
    UserIdNameStatusResponse,
    UserListResponse,
    UserModel,
    UserModelResponse,
    Users,
    UserNameResponse,
)
from open_webui.models.groups import Groups
from open_webui.models.channels import (
    Channels,
    ChannelModel,
    ChannelForm,
    ChannelResponse,
    CreateChannelForm,
)
from open_webui.models.messages import (
    Messages,
    MessageModel,
    MessageResponse,
    MessageWithReactionsResponse,
    MessageForm,
)
from open_webui.models.models import Models

from open_webui.config import ENABLE_ADMIN_CHAT_ACCESS, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS

from open_webui.utils.models import get_all_models
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import (
    has_access,
    get_users_with_access,
    get_permitted_group_and_user_ids,
    has_permission,
)
from open_webui.utils.webhook import post_webhook
from open_webui.utils.channels import (
    extract_mentions,
    replace_mentions,
    build_message_with_rag_context,
)
from open_webui.utils.misc import get_last_user_message
from open_webui.retrieval.utils import get_sources_from_items
from open_webui.routers.pipelines import (
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)
from open_webui.utils.filter import get_sorted_filter_ids, process_filter_functions
from open_webui.models.functions import Functions
from open_webui.socket.main import get_event_emitter, get_event_call

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()


async def process_message_knowledge(
    request: Request, message_content: str, knowledge_items: list[dict], user: UserModel
) -> list[dict]:
    """
    Retrieve sources from knowledge bases using simplified single-query approach.
    Optimized for channel real-time messaging (faster than multi-query generation).
    """
    if not knowledge_items or not message_content:
        return []

    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            sources = await loop.run_in_executor(
                executor,
                lambda: get_sources_from_items(
                    request=request,
                    items=knowledge_items,
                    queries=[message_content],
                    embedding_function=lambda q, p: request.app.state.EMBEDDING_FUNCTION(
                        q, prefix=p, user=user
                    ),
                    k=request.app.state.config.TOP_K,
                    reranking_function=(
                        lambda s: request.app.state.RERANKING_FUNCTION(s, user=user)
                        if request.app.state.RERANKING_FUNCTION
                        else None
                    ),
                    k_reranker=request.app.state.config.TOP_K_RERANKER,
                    r=request.app.state.config.RELEVANCE_THRESHOLD,
                    hybrid_bm25_weight=request.app.state.config.HYBRID_BM25_WEIGHT,
                    hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                    full_context=request.app.state.config.RAG_FULL_CONTEXT,
                    user=user,
                ),
            )
        return sources or []
    except Exception as e:
        log.exception(f"Error retrieving knowledge sources: {e}")
        return []


############################
# GetChatList
############################


class ChannelListItemResponse(ChannelModel):
    user_ids: Optional[list[str]] = None  # 'dm' channels only
    users: Optional[list[UserIdNameStatusResponse]] = None  # 'dm' channels only

    last_message_at: Optional[int] = None  # timestamp in epoch (time_ns)
    unread_count: int = 0


@router.get("/", response_model=list[ChannelListItemResponse])
async def get_channels(request: Request, user=Depends(get_verified_user)):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    channels = Channels.get_channels_by_user_id(user.id)
    channel_list = []
    for channel in channels:
        last_message = Messages.get_last_message_by_channel_id(channel.id)
        last_message_at = last_message.created_at if last_message else None

        channel_member = Channels.get_member_by_channel_and_user_id(channel.id, user.id)
        unread_count = (
            Messages.get_unread_message_count(
                channel.id, user.id, channel_member.last_read_at
            )
            if channel_member
            else 0
        )

        user_ids = None
        users = None
        if channel.type == "dm":
            user_ids = [
                member.user_id
                for member in Channels.get_members_by_channel_id(channel.id)
            ]
            users = [
                UserIdNameStatusResponse(
                    **{**user.model_dump(), "is_active": Users.is_user_active(user.id)}
                )
                for user in Users.get_users_by_user_ids(user_ids)
            ]

        channel_list.append(
            ChannelListItemResponse(
                **channel.model_dump(),
                user_ids=user_ids,
                users=users,
                last_message_at=last_message_at,
                unread_count=unread_count,
            )
        )

    return channel_list


@router.get("/list", response_model=list[ChannelModel])
async def get_all_channels(user=Depends(get_verified_user)):
    if user.role == "admin":
        return Channels.get_channels()
    return Channels.get_channels_by_user_id(user.id)


############################
# GetDMChannelByUserId
############################


@router.get("/users/{user_id}", response_model=Optional[ChannelModel])
async def get_dm_channel_by_user_id(
    request: Request, user_id: str, user=Depends(get_verified_user)
):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    try:
        existing_channel = Channels.get_dm_channel_by_user_ids([user.id, user_id])
        if existing_channel:
            participant_ids = [
                member.user_id
                for member in Channels.get_members_by_channel_id(existing_channel.id)
            ]

            await emit_to_users(
                "events:channel",
                {"data": {"type": "channel:created"}},
                participant_ids,
            )
            await enter_room_for_users(
                f"channel:{existing_channel.id}", participant_ids
            )

            Channels.update_member_active_status(existing_channel.id, user.id, True)
            return ChannelModel(**existing_channel.model_dump())

        channel = Channels.insert_new_channel(
            CreateChannelForm(
                type="dm",
                name="",
                user_ids=[user_id],
            ),
            user.id,
        )

        if channel:
            participant_ids = [
                member.user_id
                for member in Channels.get_members_by_channel_id(channel.id)
            ]

            await emit_to_users(
                "events:channel",
                {"data": {"type": "channel:created"}},
                participant_ids,
            )
            await enter_room_for_users(f"channel:{channel.id}", participant_ids)

            return ChannelModel(**channel.model_dump())
        else:
            raise Exception("Error creating channel")
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# CreateNewChannel
############################


@router.post("/create", response_model=Optional[ChannelModel])
async def create_new_channel(
    request: Request, form_data: CreateChannelForm, user=Depends(get_verified_user)
):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if form_data.type not in ["group", "dm"] and user.role != "admin":
        # Only admins can create standard channels (joined by default)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    try:
        if form_data.type == "dm":
            existing_channel = Channels.get_dm_channel_by_user_ids(
                [user.id, *form_data.user_ids]
            )
            if existing_channel:
                participant_ids = [
                    member.user_id
                    for member in Channels.get_members_by_channel_id(
                        existing_channel.id
                    )
                ]
                await emit_to_users(
                    "events:channel",
                    {"data": {"type": "channel:created"}},
                    participant_ids,
                )
                await enter_room_for_users(
                    f"channel:{existing_channel.id}", participant_ids
                )

                Channels.update_member_active_status(existing_channel.id, user.id, True)
                return ChannelModel(**existing_channel.model_dump())

        channel = Channels.insert_new_channel(form_data, user.id)

        if channel:
            participant_ids = [
                member.user_id
                for member in Channels.get_members_by_channel_id(channel.id)
            ]

            await emit_to_users(
                "events:channel",
                {"data": {"type": "channel:created"}},
                participant_ids,
            )
            await enter_room_for_users(f"channel:{channel.id}", participant_ids)

            return ChannelModel(**channel.model_dump())
        else:
            raise Exception("Error creating channel")
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChannelById
############################


class ChannelFullResponse(ChannelResponse):
    user_ids: Optional[list[str]] = None  # 'group'/'dm' channels only
    users: Optional[list[UserIdNameStatusResponse]] = None  # 'group'/'dm' channels only

    last_read_at: Optional[int] = None  # timestamp in epoch (time_ns)
    unread_count: int = 0


@router.get("/{id}", response_model=Optional[ChannelFullResponse])
async def get_channel_by_id(id: str, user=Depends(get_verified_user)):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    user_ids = None
    users = None

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

        user_ids = [
            member.user_id for member in Channels.get_members_by_channel_id(channel.id)
        ]

        users = [
            UserIdNameStatusResponse(
                **{**user.model_dump(), "is_active": Users.is_user_active(user.id)}
            )
            for user in Users.get_users_by_user_ids(user_ids)
        ]

        channel_member = Channels.get_member_by_channel_and_user_id(channel.id, user.id)
        unread_count = Messages.get_unread_message_count(
            channel.id, user.id, channel_member.last_read_at if channel_member else None
        )

        return ChannelFullResponse(
            **{
                **channel.model_dump(),
                "user_ids": user_ids,
                "users": users,
                "is_manager": Channels.is_user_channel_manager(channel.id, user.id),
                "write_access": True,
                "user_count": len(user_ids),
                "last_read_at": channel_member.last_read_at if channel_member else None,
                "unread_count": unread_count,
            }
        )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="read", access_control=channel.access_control
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

        write_access = has_access(
            user.id, type="write", access_control=channel.access_control, strict=False
        )

        user_count = len(get_users_with_access("read", channel.access_control))

        channel_member = Channels.get_member_by_channel_and_user_id(channel.id, user.id)
        unread_count = Messages.get_unread_message_count(
            channel.id, user.id, channel_member.last_read_at if channel_member else None
        )

        return ChannelFullResponse(
            **{
                **channel.model_dump(),
                "user_ids": user_ids,
                "users": users,
                "is_manager": Channels.is_user_channel_manager(channel.id, user.id),
                "write_access": write_access or user.role == "admin",
                "user_count": user_count,
                "last_read_at": channel_member.last_read_at if channel_member else None,
                "unread_count": unread_count,
            }
        )


############################
# GetChannelMembersById
############################


PAGE_ITEM_COUNT = 30


@router.get("/{id}/members", response_model=UserListResponse)
async def get_channel_members_by_id(
    id: str,
    query: Optional[str] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
):

    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    limit = PAGE_ITEM_COUNT

    page = max(1, page)
    skip = (page - 1) * limit

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    if channel.type == "dm":
        user_ids = [
            member.user_id for member in Channels.get_members_by_channel_id(channel.id)
        ]
        users = Users.get_users_by_user_ids(user_ids)
        total = len(users)

        return {
            "users": [
                UserModelResponse(
                    **user.model_dump(), is_active=Users.is_user_active(user.id)
                )
                for user in users
            ],
            "total": total,
        }
    else:
        filter = {}

        if query:
            filter["query"] = query
        if order_by:
            filter["order_by"] = order_by
        if direction:
            filter["direction"] = direction

        if channel.type == "group":
            filter["channel_id"] = channel.id
        else:
            filter["roles"] = ["!pending"]
            permitted_ids = get_permitted_group_and_user_ids(
                "read", channel.access_control
            )
            if permitted_ids:
                filter["user_ids"] = permitted_ids.get("user_ids")
                filter["group_ids"] = permitted_ids.get("group_ids")

        result = Users.get_users(filter=filter, skip=skip, limit=limit)

        users = result["users"]
        total = result["total"]

        return {
            "users": [
                UserModelResponse(
                    **user.model_dump(), is_active=Users.is_user_active(user.id)
                )
                for user in users
            ],
            "total": total,
        }


#################################################
# UpdateIsActiveMemberByIdAndUserId
#################################################


class UpdateActiveMemberForm(BaseModel):
    is_active: bool


@router.post("/{id}/members/active", response_model=bool)
async def update_is_active_member_by_id_and_user_id(
    id: str,
    form_data: UpdateActiveMemberForm,
    user=Depends(get_verified_user),
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if not Channels.is_user_channel_member(channel.id, user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    Channels.update_member_active_status(channel.id, user.id, form_data.is_active)
    return True


#################################################
# AddMembersById
#################################################


class UpdateMembersForm(BaseModel):
    user_ids: list[str] = []
    group_ids: list[str] = []


@router.post("/{id}/update/members/add")
async def add_members_by_id(
    request: Request,
    id: str,
    form_data: UpdateMembersForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.user_id != user.id and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        memberships = Channels.add_members_to_channel(
            channel.id, user.id, form_data.user_ids, form_data.group_ids
        )

        return memberships
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


#################################################
#
#################################################


class RemoveMembersForm(BaseModel):
    user_ids: list[str] = []


@router.post("/{id}/update/members/remove")
async def remove_members_by_id(
    request: Request,
    id: str,
    form_data: RemoveMembersForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.user_id != user.id and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        deleted = Channels.remove_members_from_channel(channel.id, form_data.user_ids)

        return deleted
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UpdateChannelById
############################


@router.post("/{id}/update", response_model=Optional[ChannelModel])
async def update_channel_by_id(
    request: Request, id: str, form_data: ChannelForm, user=Depends(get_verified_user)
):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.user_id != user.id and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        channel = Channels.update_channel_by_id(id, form_data)
        return ChannelModel(**channel.model_dump())
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteChannelById
############################


@router.delete("/{id}/delete", response_model=bool)
async def delete_channel_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    if user.role != "admin" and not has_permission(
        user.id, "features.channels", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.user_id != user.id and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        Channels.delete_channel_by_id(id)
        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChannelMessages
############################


class MessageUserResponse(MessageResponse):
    pass


@router.get("/{id}/messages", response_model=list[MessageUserResponse])
async def get_channel_messages(
    id: str, skip: int = 0, limit: int = 50, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="read", access_control=channel.access_control
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

        channel_member = Channels.join_channel(
            id, user.id
        )  # Ensure user is a member of the channel

    message_list = Messages.get_messages_by_channel_id(id, skip, limit)
    users = {}

    messages = []
    for message in message_list:
        if message.user_id not in users:
            user = Users.get_user_by_id(message.user_id)
            users[message.user_id] = user

        thread_replies = Messages.get_thread_replies_by_message_id(message.id)
        latest_thread_reply_at = (
            thread_replies[0].created_at if thread_replies else None
        )

        messages.append(
            MessageUserResponse(
                **{
                    **message.model_dump(),
                    "reply_count": len(thread_replies),
                    "latest_reply_at": latest_thread_reply_at,
                    "reactions": Messages.get_reactions_by_message_id(message.id),
                    "user": UserNameResponse(**users[message.user_id].model_dump()),
                }
            )
        )

    return messages


############################
# GetPinnedChannelMessages
############################

PAGE_ITEM_COUNT_PINNED = 20


@router.get("/{id}/messages/pinned", response_model=list[MessageWithReactionsResponse])
async def get_pinned_channel_messages(
    id: str, page: int = 1, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="read", access_control=channel.access_control
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    page = max(1, page)
    skip = (page - 1) * PAGE_ITEM_COUNT_PINNED
    limit = PAGE_ITEM_COUNT_PINNED

    message_list = Messages.get_pinned_messages_by_channel_id(id, skip, limit)
    users = {}

    messages = []
    for message in message_list:
        if message.user_id not in users:
            user = Users.get_user_by_id(message.user_id)
            users[message.user_id] = user

        messages.append(
            MessageWithReactionsResponse(
                **{
                    **message.model_dump(),
                    "reactions": Messages.get_reactions_by_message_id(message.id),
                    "user": UserNameResponse(**users[message.user_id].model_dump()),
                }
            )
        )

    return messages


############################
# PostNewMessage
############################


async def send_notification(name, webui_url, channel, message, active_user_ids):
    users = get_users_with_access("read", channel.access_control)

    for user in users:
        if (user.id not in active_user_ids) and Channels.is_user_channel_member(
            channel.id, user.id
        ):
            if user.settings:
                webhook_url = user.settings.ui.get("notifications", {}).get(
                    "webhook_url", None
                )
                if webhook_url:
                    await post_webhook(
                        name,
                        webhook_url,
                        f"#{channel.name} - {webui_url}/channels/{channel.id}\n\n{message.content}",
                        {
                            "action": "channel",
                            "message": message.content,
                            "title": channel.name,
                            "url": f"{webui_url}/channels/{channel.id}",
                        },
                    )

    return True


async def model_response_handler(request, channel, message, user):
    # Ensure models are loaded in app state (same as chat interface)
    # Force refresh to ensure filters are loaded (they come from OpenAI API)
    await get_all_models(request, refresh=True, user=user)
    
    # Use app.state.MODELS directly (includes all models including filters)
    # This matches how the chat interface loads models
    MODELS = request.app.state.MODELS
    
    log.info(f"Channel: Model response handler called for channel {channel.id}, message {message.id}")

    mentions = extract_mentions(message.content)
    message_content = replace_mentions(message.content)

    model_mentions = {}

    # check if the message is a reply to a message sent by a model
    if (
        message.reply_to_message
        and message.reply_to_message.meta
        and message.reply_to_message.meta.get("model_id", None)
    ):
        model_id = message.reply_to_message.meta.get("model_id", None)
        model_mentions[model_id] = {"id": model_id, "id_type": "M"}

    # check if any of the mentions are models
    for mention in mentions:
        if mention["id_type"] == "M" and mention["id"] not in model_mentions:
            model_mentions[mention["id"]] = mention

    if not model_mentions:
        return False

    for mention in model_mentions.values():
        model_id = mention["id"]
        model = MODELS.get(model_id, None)

        if model:
            # CRITICAL: Merge pipeline config from database if model doesn't have it
            # Pipeline config is stored in model_info.params.pipeline but params are
            # removed from model["info"] during get_all_models() to avoid exposing sensitive info
            # So we need to merge it from database at runtime
            # NOTE: model is a reference to MODELS[model_id], so modifying it updates MODELS automatically
            if "pipeline" not in model:
                log.debug(f"Channel: Model {model_id} has no pipeline field, checking database...")
                model_info = Models.get_model_by_id(model_id)
                if model_info and model_info.params:
                    params = model_info.params.model_dump() if hasattr(model_info.params, 'model_dump') else model_info.params
                    if isinstance(params, dict) and "pipeline" in params:
                        model["pipeline"] = params["pipeline"]
                        # model is already a reference to MODELS[model_id], so MODELS is automatically updated
                        log.debug(f"Channel: Merged pipeline config from database for model {model_id}")
            
            # Re-fetch model from MODELS to ensure we have latest (in case it was updated)
            model = MODELS.get(model_id, None)
            
            try:
                # reverse to get in chronological order
                thread_messages = Messages.get_messages_by_parent_id(
                    channel.id,
                    message.parent_id if message.parent_id else message.id,
                )[::-1]

                response_message, channel = await new_message_handler(
                    request,
                    channel.id,
                    MessageForm(
                        **{
                            "parent_id": (
                                message.parent_id if message.parent_id else message.id
                            ),
                            "content": f"",
                            "data": {},
                            "meta": {
                                "model_id": model_id,
                                "model_name": model.get("name", model_id),
                            },
                        }
                    ),
                    user,
                )

                thread_history = []
                images = []
                message_users = {}

                for thread_message in thread_messages:
                    message_user = None
                    if thread_message.user_id not in message_users:
                        message_user = Users.get_user_by_id(thread_message.user_id)
                        message_users[thread_message.user_id] = message_user
                    else:
                        message_user = message_users[thread_message.user_id]

                    if thread_message.meta and thread_message.meta.get(
                        "model_id", None
                    ):
                        # If the message was sent by a model, use the model name
                        message_model_id = thread_message.meta.get("model_id", None)
                        message_model = MODELS.get(message_model_id, None)
                        username = (
                            message_model.get("name", message_model_id)
                            if message_model
                            else message_model_id
                        )
                    else:
                        username = message_user.name if message_user else "Unknown"

                    thread_history.append(
                        f"{username}: {replace_mentions(thread_message.content)}"
                    )

                    thread_message_files = thread_message.data.get("files", [])
                    for file in thread_message_files:
                        if file.get("type", "") == "image":
                            images.append(file.get("url", ""))

                thread_history_string = "\n\n".join(thread_history)
                system_message = {
                    "role": "system",
                    "content": f"You are {model.get('name', model_id)}, participating in a threaded conversation. Be concise and conversational."
                    + (
                        f"Here's the thread history:\n\n\n{thread_history_string}\n\n\nContinue the conversation naturally as {model.get('name', model_id)}, addressing the most recent message while being aware of the full context."
                        if thread_history
                        else ""
                    ),
                }

                # Check if message has knowledge sources and inject RAG context
                content = message_content
                if message.meta and message.meta.get("sources"):
                    # Inject knowledge context using RAG template
                    rag_template_str = request.app.state.config.RAG_TEMPLATE
                    content = build_message_with_rag_context(
                        message_content, message.meta["sources"], rag_template_str
                    )
                    log.debug(f"Injected RAG context from {len(message.meta['sources'])} sources")

                content = f"{user.name if user else 'User'}: {content}"
                
                if images:
                    content = [
                        {
                            "type": "text",
                            "text": content,
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image,
                                },
                            }
                            for image in images
                        ],
                    ]

                form_data = {
                    "model": model_id,
                    "messages": [
                        system_message,
                        {"role": "user", "content": content},
                    ],
                    "stream": False,
                }

                # Process through inlet filters (pre-processing)
                log.info(f"Channel: Processing inlet filters for model {model_id}")
                
                # STEP 1: Pipeline filters (external filter services)
                from open_webui.routers.pipelines import get_sorted_filters
                sorted_filters = get_sorted_filters(model_id, MODELS)
                
                # Check if the model itself has a pipeline field
                model_has_pipeline = "pipeline" in model
                if model_has_pipeline:
                    # Note: process_pipeline_inlet_filter will add model to sorted_filters if it has pipeline
                    sorted_filters_with_model = sorted_filters + [model]
                else:
                    sorted_filters_with_model = sorted_filters
                
                if sorted_filters_with_model:
                    log.info(f"Channel: Found {len(sorted_filters_with_model)} pipeline inlet filter(s) for model {model_id}: {[f.get('id') for f in sorted_filters_with_model]}")
                else:
                    log.debug(f"Channel: No pipeline inlet filters configured for model {model_id}")
                
                try:
                    original_model = form_data["model"]
                    original_messages = form_data.get("messages")
                    form_data = await process_pipeline_inlet_filter(
                        request, form_data, user, MODELS
                    )
                    if form_data["model"] != original_model or form_data.get("messages") != original_messages:
                        log.info(f"Channel: ✓ Pipeline inlet filters modified request for model {model_id}")
                except Exception as e:
                    log.error(f"Channel: Pipeline inlet filter processing failed: {e}")
                    # Continue with unprocessed request
                
                # STEP 2: Function filters (database-stored filter functions like chat summarization/re-rank)
                # Create metadata for function filters (similar to chat but for channels)
                # NOTE: Filters expect 'chat_id' in metadata, so we pass channel.id as chat_id
                metadata = {
                    "user_id": user.id,
                    "chat_id": channel.id,  # Filters expect chat_id, so use channel.id
                    "channel_id": channel.id,  # Also keep channel_id for channel-specific logic
                    "message_id": response_message.id,
                    "filter_ids": [],  # Can be populated from message metadata if needed
                }
                
                # Create extra_params for function filters
                event_emitter = get_event_emitter(metadata, update_db=False)  # Don't update DB for channels
                event_call = get_event_call(metadata)
                
                extra_params = {
                    "__event_emitter__": event_emitter,
                    "__event_call__": event_call,
                    "__user__": user.model_dump() if isinstance(user, UserModel) else {},
                    "__metadata__": metadata,
                    "__request__": request,
                    "__model__": model,
                }
                
                log.info(f"Channel: Processing function inlet filters for model {model_id}")
                try:
                    # Get function filter IDs configured for this model
                    filter_ids = get_sorted_filter_ids(
                        request, model, metadata.get("filter_ids", [])
                    )
                    if filter_ids:
                        log.info(f"Channel: Found {len(filter_ids)} function inlet filter(s) for model {model_id}: {filter_ids}")
                        filter_functions = [
                            Functions.get_function_by_id(filter_id)
                            for filter_id in filter_ids
                        ]
                        
                        form_data, flags = await process_filter_functions(
                            request=request,
                            filter_functions=filter_functions,
                            filter_type="inlet",
                            form_data=form_data,
                            extra_params=extra_params,
                        )
                        log.info(f"Channel: Function inlet filters processed for model {model_id}, skip_files={flags.get('skip_files') if flags else None}")
                    else:
                        log.debug(f"Channel: No function inlet filters configured for model {model_id}")
                except Exception as e:
                    log.error(f"Channel: Function inlet filter processing failed: {e}")
                    # Continue with unprocessed request

                res = await generate_chat_completion(
                    request,
                    form_data=form_data,
                    user=user,
                )

                if res:
                    # Process through outlet filters (summary, re-rank, etc.)
                    # STEP 1: Pipeline outlet filters
                    log.debug(f"Channel: Processing pipeline outlet filters for model {model_id}")
                    try:
                        original_content = res.get("choices", [{}])[0].get("message", {}).get("content") if res.get("choices") else None
                        res = await process_pipeline_outlet_filter(
                            request, res, user, MODELS
                        )
                        new_content = res.get("choices", [{}])[0].get("message", {}).get("content") if res.get("choices") else None
                        if original_content and new_content and original_content != new_content:
                            log.info(f"Channel: ✓ Pipeline outlet filters modified response for model {model_id}")
                    except Exception as e:
                        log.error(f"Channel: Pipeline outlet filter processing failed: {e}")
                        # Continue with unprocessed response
                    
                    # STEP 2: Function outlet filters (chat summarization, re-rank, etc.)
                    log.info(f"Channel: Processing function outlet filters for model {model_id}")
                    try:
                        # Convert response to format expected by function filters
                        # Function filters expect form_data format similar to chat_completed
                        filter_form_data = {
                            "id": response_message.id,
                            "model": model_id,
                            "choices": res.get("choices", []),
                            "chat_id": channel.id,  # Filters expect chat_id, use channel.id for channels
                            "message_id": response_message.id,
                            "session_id": None,
                            "user_id": user.id,
                            "filter_ids": metadata.get("filter_ids", []),
                        }
                        
                        filter_ids = get_sorted_filter_ids(
                            request, model, metadata.get("filter_ids", [])
                        )
                        
                        if filter_ids:
                            log.info(f"Channel: Found {len(filter_ids)} function outlet filter(s) for model {model_id}: {filter_ids}")
                            filter_functions = [
                                Functions.get_function_by_id(filter_id)
                                for filter_id in filter_ids
                            ]
                            
                            original_content = filter_form_data.get("choices", [{}])[0].get("message", {}).get("content") if filter_form_data.get("choices") else None
                            
                            result, _ = await process_filter_functions(
                                request=request,
                                filter_functions=filter_functions,
                                filter_type="outlet",
                                form_data=filter_form_data,
                                extra_params=extra_params,
                            )
                            
                            # Update res with filtered result
                            if result and result.get("choices"):
                                res["choices"] = result["choices"]
                                new_content = result["choices"][0].get("message", {}).get("content") if result["choices"] else None
                                if original_content != new_content:
                                    log.info(f"Channel: ✓ Function outlet filters modified response for model {model_id}")
                                else:
                                    log.info(f"Channel: Function outlet filters processed but no content change for model {model_id}")
                            else:
                                log.warning(f"Channel: Function outlet filters returned no choices for model {model_id}")
                        else:
                            log.debug(f"Channel: No function outlet filters configured for model {model_id}")
                    except Exception as e:
                        log.error(f"Channel: Function outlet filter processing failed: {e}")
                        log.exception(e)
                        # Continue with unprocessed response

                    if res.get("choices", []) and len(res["choices"]) > 0:
                        await update_message_by_id(
                            channel.id,
                            response_message.id,
                            MessageForm(
                                **{
                                    "content": res["choices"][0]["message"]["content"],
                                    "meta": {
                                        "done": True,
                                    },
                                }
                            ),
                            user,
                        )
                    elif res.get("error", None):
                        await update_message_by_id(
                            channel.id,
                            response_message.id,
                            MessageForm(
                                **{
                                    "content": f"Error: {res['error']}",
                                    "meta": {
                                        "done": True,
                                    },
                                }
                            ),
                            user,
                        )
            except Exception as e:
                log.info(e)
                pass

    return True


async def new_message_handler(
    request: Request, id: str, form_data: MessageForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="write", access_control=channel.access_control, strict=False
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    try:
        # Extract knowledge bases from message data
        knowledge_items = []
        if form_data.data and form_data.data.get("files"):
            knowledge_items = [
                f
                for f in form_data.data["files"]
                if f.get("type") == "collection" or f.get("knowledge")
            ]

        # Process knowledge for RAG retrieval
        if knowledge_items:
            sources = await process_message_knowledge(
                request, form_data.content, knowledge_items, user
            )
            
            if sources:
                form_data.meta = form_data.meta or {}
                form_data.meta["sources"] = sources
                log.debug(f"Retrieved {len(sources)} sources from {len(knowledge_items)} knowledge bases")

        # Insert message (sources already in meta)
        message = Messages.insert_new_message(form_data, channel.id, user.id)
        if message:
            if channel.type in ["group", "dm"]:
                members = Channels.get_members_by_channel_id(channel.id)
                for member in members:
                    if not member.is_active:
                        Channels.update_member_active_status(
                            channel.id, member.user_id, True
                        )

            message = Messages.get_message_by_id(message.id)

            # Emit message event
            await sio.emit(
                "events:channel",
                {
                    "channel_id": channel.id,
                    "message_id": message.id,
                    "data": {
                        "type": "message",
                        "data": message.model_dump(),
                    },
                    "user": UserNameResponse(**user.model_dump()).model_dump(),
                    "channel": channel.model_dump(),
                },
                to=f"channel:{channel.id}",
            )

            if message.parent_id:
                # If this message is a reply, emit to the parent message as well
                parent_message = Messages.get_message_by_id(message.parent_id)

                if parent_message:
                    await sio.emit(
                        "events:channel",
                        {
                            "channel_id": channel.id,
                            "message_id": parent_message.id,
                            "data": {
                                "type": "message:reply",
                                "data": parent_message.model_dump(),
                            },
                            "user": UserNameResponse(**user.model_dump()).model_dump(),
                            "channel": channel.model_dump(),
                        },
                        to=f"channel:{channel.id}",
                    )
            return message, channel
        else:
            raise Exception("Error creating message")
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


@router.post("/{id}/messages/post", response_model=Optional[MessageModel])
async def post_new_message(
    request: Request,
    id: str,
    form_data: MessageForm,
    background_tasks: BackgroundTasks,
    user=Depends(get_verified_user),
):

    try:
        message, channel = await new_message_handler(request, id, form_data, user)
        active_user_ids = get_user_ids_from_room(f"channel:{channel.id}")

        async def background_handler():
            await model_response_handler(request, channel, message, user)
            await send_notification(
                request.app.state.WEBUI_NAME,
                request.app.state.config.WEBUI_URL,
                channel,
                message,
                active_user_ids,
            )

        background_tasks.add_task(background_handler)

        return message

    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChannelMessage
############################


@router.get("/{id}/messages/{message_id}", response_model=Optional[MessageUserResponse])
async def get_channel_message(
    id: str, message_id: str, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="read", access_control=channel.access_control
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    return MessageUserResponse(
        **{
            **message.model_dump(),
            "user": UserNameResponse(
                **Users.get_user_by_id(message.user_id).model_dump()
            ),
        }
    )


############################
# PinChannelMessage
############################


class PinMessageForm(BaseModel):
    is_pinned: bool


@router.post(
    "/{id}/messages/{message_id}/pin", response_model=Optional[MessageUserResponse]
)
async def pin_channel_message(
    id: str, message_id: str, form_data: PinMessageForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="read", access_control=channel.access_control
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        Messages.update_is_pinned_by_id(message_id, form_data.is_pinned, user.id)
        message = Messages.get_message_by_id(message_id)
        return MessageUserResponse(
            **{
                **message.model_dump(),
                "user": UserNameResponse(
                    **Users.get_user_by_id(message.user_id).model_dump()
                ),
            }
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChannelThreadMessages
############################


@router.get(
    "/{id}/messages/{message_id}/thread", response_model=list[MessageUserResponse]
)
async def get_channel_thread_messages(
    id: str,
    message_id: str,
    skip: int = 0,
    limit: int = 50,
    user=Depends(get_verified_user),
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="read", access_control=channel.access_control
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    message_list = Messages.get_messages_by_parent_id(id, message_id, skip, limit)
    users = {}

    messages = []
    for message in message_list:
        if message.user_id not in users:
            user = Users.get_user_by_id(message.user_id)
            users[message.user_id] = user

        messages.append(
            MessageUserResponse(
                **{
                    **message.model_dump(),
                    "reply_count": 0,
                    "latest_reply_at": None,
                    "reactions": Messages.get_reactions_by_message_id(message.id),
                    "user": UserNameResponse(**users[message.user_id].model_dump()),
                }
            )
        )

    return messages


############################
# UpdateMessageById
############################


@router.post(
    "/{id}/messages/{message_id}/update", response_model=Optional[MessageModel]
)
async def update_message_by_id(
    id: str, message_id: str, form_data: MessageForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if (
            user.role != "admin"
            and message.user_id != user.id
            and not has_access(
                user.id, type="read", access_control=channel.access_control
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    try:
        message = Messages.update_message_by_id(message_id, form_data)
        message = Messages.get_message_by_id(message_id)

        if message:
            await sio.emit(
                "events:channel",
                {
                    "channel_id": channel.id,
                    "message_id": message.id,
                    "data": {
                        "type": "message:update",
                        "data": message.model_dump(),
                    },
                    "user": UserNameResponse(**user.model_dump()).model_dump(),
                    "channel": channel.model_dump(),
                },
                to=f"channel:{channel.id}",
            )

        return MessageModel(**message.model_dump())
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# AddReactionToMessage
############################


class ReactionForm(BaseModel):
    name: str


@router.post("/{id}/messages/{message_id}/reactions/add", response_model=bool)
async def add_reaction_to_message(
    id: str, message_id: str, form_data: ReactionForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="write", access_control=channel.access_control, strict=False
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        Messages.add_reaction_to_message(message_id, user.id, form_data.name)
        message = Messages.get_message_by_id(message_id)

        await sio.emit(
            "events:channel",
            {
                "channel_id": channel.id,
                "message_id": message.id,
                "data": {
                    "type": "message:reaction:add",
                    "data": {
                        **message.model_dump(),
                        "name": form_data.name,
                    },
                },
                "user": UserNameResponse(**user.model_dump()).model_dump(),
                "channel": channel.model_dump(),
            },
            to=f"channel:{channel.id}",
        )

        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# RemoveReactionById
############################


@router.post("/{id}/messages/{message_id}/reactions/remove", response_model=bool)
async def remove_reaction_by_id_and_user_id_and_name(
    id: str, message_id: str, form_data: ReactionForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if user.role != "admin" and not has_access(
            user.id, type="write", access_control=channel.access_control, strict=False
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    try:
        Messages.remove_reaction_by_id_and_user_id_and_name(
            message_id, user.id, form_data.name
        )

        message = Messages.get_message_by_id(message_id)

        await sio.emit(
            "events:channel",
            {
                "channel_id": channel.id,
                "message_id": message.id,
                "data": {
                    "type": "message:reaction:remove",
                    "data": {
                        **message.model_dump(),
                        "name": form_data.name,
                    },
                },
                "user": UserNameResponse(**user.model_dump()).model_dump(),
                "channel": channel.model_dump(),
            },
            to=f"channel:{channel.id}",
        )

        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteMessageById
############################


@router.delete("/{id}/messages/{message_id}/delete", response_model=bool)
async def delete_message_by_id(
    id: str, message_id: str, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    if channel.type in ["group", "dm"]:
        if not Channels.is_user_channel_member(channel.id, user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )
    else:
        if (
            user.role != "admin"
            and message.user_id != user.id
            and not has_access(
                user.id,
                type="write",
                access_control=channel.access_control,
                strict=False,
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.DEFAULT()
            )

    try:
        Messages.delete_message_by_id(message_id)
        await sio.emit(
            "events:channel",
            {
                "channel_id": channel.id,
                "message_id": message.id,
                "data": {
                    "type": "message:delete",
                    "data": {
                        **message.model_dump(),
                        "user": UserNameResponse(**user.model_dump()).model_dump(),
                    },
                },
                "user": UserNameResponse(**user.model_dump()).model_dump(),
                "channel": channel.model_dump(),
            },
            to=f"channel:{channel.id}",
        )

        if message.parent_id:
            # If this message is a reply, emit to the parent message as well
            parent_message = Messages.get_message_by_id(message.parent_id)

            if parent_message:
                await sio.emit(
                    "events:channel",
                    {
                        "channel_id": channel.id,
                        "message_id": parent_message.id,
                        "data": {
                            "type": "message:reply",
                            "data": parent_message.model_dump(),
                        },
                        "user": UserNameResponse(**user.model_dump()).model_dump(),
                        "channel": channel.model_dump(),
                    },
                    to=f"channel:{channel.id}",
                )

        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )
