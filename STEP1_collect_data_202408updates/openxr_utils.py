import ctypes
from ctypes import cast, byref, POINTER
import time
import xr


class ContextObject(object):
    def __init__(
            self,
            instance_create_info: xr.InstanceCreateInfo = xr.InstanceCreateInfo(),
            session_create_info: xr.SessionCreateInfo = xr.SessionCreateInfo(),
            reference_space_create_info: xr.ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo(),
            view_configuration_type: xr.ViewConfigurationType = xr.ViewConfigurationType.PRIMARY_STEREO,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
            use_session=True  # ✅ 新增：控制是否创建session
    ):
        self._instance_create_info = instance_create_info
        self.instance = None

        self._session_create_info = session_create_info
        self.session = None
        self.space = None

        self.use_session = use_session  # ✅ 核心开关

        self.session_state = xr.SessionState.IDLE
        self._reference_space_create_info = reference_space_create_info
        self.view_configuration_type = view_configuration_type
        self.environment_blend_mode = environment_blend_mode
        self.form_factor = form_factor

        self.graphics = None
        self.graphics_binding_pointer = None

        self.action_sets = []
        self.default_action_set = None

        self.render_layers = []
        self.swapchains = []
        self.swapchain_image_ptr_buffers = []
        self.swapchain_image_buffers = []

        self.exit_render_loop = False
        self.request_restart = False
        self.session_is_running = False

    def __enter__(self):
        # ✅ 创建 instance（始终可以）
        self.instance = xr.create_instance(
            create_info=self._instance_create_info,
        )

        # ✅ 获取 system（某些 runtime 可能仍失败，但一般OK）
        try:
            self.system_id = xr.get_system(
                instance=self.instance,
                get_info=xr.SystemGetInfo(
                    form_factor=self.form_factor,
                ),
            )
        except Exception as e:
            print("[WARN] xr.get_system failed (no HMD?):", e)
            self.system_id = None

        # ✅ 如果不开 session → 到此为止
        if not self.use_session or self.system_id is None:
            print("[INFO] Running in HEADLESS mode (no XR session)")
            self.session = None
            self.space = None
            return self

        # =============================
        # 正常 XR 模式（有头显才会进）
        # =============================
        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next

        self._session_create_info.system_id = self.system_id

        self.session = xr.create_session(
            instance=self.instance,
            create_info=self._session_create_info,
        )

        self.space = xr.create_reference_space(
            session=self.session,
            create_info=self._reference_space_create_info
        )

        # ✅ action set（仅在有 session 时）
        self.default_action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        self.action_sets.append(self.default_action_set)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None

        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None

        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None

        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None

        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    # ❌ 禁用 frame_loop（无头模式不能用）
    def frame_loop(self):
        if not self.use_session or self.session is None:
            raise RuntimeError("frame_loop() not available in headless mode")

        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(
                    *self.action_sets
                )
            ),
        )

        while True:
            self.exit_render_loop = False
            self.poll_xr_events()

            if self.exit_render_loop:
                break

            if self.session_is_running:
                frame_state = xr.wait_frame(self.session)
                xr.begin_frame(self.session)

                self.render_layers = []
                yield frame_state

                xr.end_frame(
                    self.session,
                    frame_end_info=xr.FrameEndInfo(
                        display_time=frame_state.predicted_display_time,
                        environment_blend_mode=self.environment_blend_mode,
                        layers=self.render_layers,
                    )
                )
            else:
                time.sleep(0.25)

    def poll_xr_events(self):
        if self.instance is None:
            return

        while True:
            try:
                event_buffer = xr.poll_event(self.instance)

                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError:
                    continue

                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED and self.session:
                    event = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataSessionStateChanged)
                    ).contents

                    self.session_state = xr.SessionState(event.state)

                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(
                            session=self.session,
                            begin_info=xr.SessionBeginInfo(
                                self.view_configuration_type,
                            ),
                        )
                        self.session_is_running = True

                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)

                elif event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                    vive_tracker_connected = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataViveTrackerConnectedHTCX)
                    ).contents

                    paths = vive_tracker_connected.paths.contents
                    persistent_path_str = xr.path_to_string(
                        self.instance,
                        paths.persistent_path
                    )

                    print(f"[INFO] Vive Tracker connected: {persistent_path_str}")

            except xr.EventUnavailable:
                break