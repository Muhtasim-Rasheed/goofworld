use macroquad::prelude::*;
use ::rand::{rng, Rng};
use tobj;

fn deg2rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

fn draw_text_3d(camera: &FreeLookCamera, text: &str, world_pos: Vec3, projection: Mat4, base_scale: f32, color: Color) {
    let screen_size = vec2(screen_width(), screen_height());

    // Convert world coordinates to 2D screen coordinates
    let screen_pos = camera.world_to_screen(world_pos, projection, screen_size);

    // Scale text based on depth (optional for depth effect)
    let depth = (camera.position - world_pos).length();
    // let scale = (1.0 / depth).clamp(0.5, 3.0); // Prevents text from getting too small/large
    let scale = (base_scale / depth).clamp(0.001, 5.0);

    let text_width = measure_text(text, None, 20, scale).width;
    let text_height = measure_text(text, None, 20, scale).height;
    
    if camera.is_world_in_screen(world_pos, projection) {
        draw_text(text, screen_pos.x - text_width / 2., screen_pos.y - text_height / 2., 20.0 * scale, color);
    }
}

fn load_model(path: &str, offset: (f32, f32, f32), color: Color) -> Vec<Mesh> {
    let mut meshes = Vec::new();
    let (models, _) = tobj::load_obj(path, &tobj::LoadOptions::default()).unwrap();
    
    for model in models {
        let mesh = &model.mesh;

        let mut vertices: Vec<Vertex> = Vec::new();
        for i in 0..mesh.positions.len() / 3 {
            let pos = vec3(
                mesh.positions[i * 3] + offset.0,
                mesh.positions[i * 3 + 1] + offset.1,
                mesh.positions[i * 3 + 2] + offset.2,
            );

            let normal = if mesh.normals.len() >= (i * 3 + 3) {
                vec3(
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                )
            } else {
                vec3(0.0, 0.0, 1.0) // Default normal
            };

            let uv = if mesh.texcoords.len() >= (i * 2 + 2) {
                vec2(mesh.texcoords[i * 2], 1.0 - mesh.texcoords[i * 2 + 1]) // Flip V-axis
            } else {
                vec2(0.0, 0.0) // Default UV
            };

            vertices.push(Vertex {
                position: pos,
                normal: vec4(normal.x, normal.y, normal.z, 0.0),
                uv,
                color: color.into()
            });
        }

        let indices: Vec<u16> = mesh.indices.iter().map(|&i| i as u16).collect();

        meshes.push(Mesh {
            vertices,
            indices,
            texture: None, // You can load a texture here if needed
        });
    }

    meshes
}

fn random_string(len: usize) -> String {
    let mut rng = rng();
    let mut s = String::new();
    let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*()`-_=+[{]}\\|;:'\",<.>/?";
    for _ in 0..len {
        let idx = rng.random_range(0..chars.len());
        s.push(chars.chars().nth(idx).unwrap());
    }
    s
}

struct FreeLookCamera {
    position: Vec3,
    yaw: f32,   // Left/Right rotation (horizontal)
    pitch: f32, // Up/Down rotation (vertical)
    last_mouse_pos: Vec2,
}

impl FreeLookCamera {
    fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            last_mouse_pos: mouse_position().into(),
        }
    }

    fn get_target(&self) -> Vec3 {
        // Convert yaw & pitch angles into a directional vector
        let dir_x = self.yaw.cos() * self.pitch.cos();
        let dir_y = self.pitch.sin();
        let dir_z = self.yaw.sin() * self.pitch.cos();

        self.position + vec3(dir_x, dir_y, dir_z)
    }

    fn world_to_screen(&self, world_pos: Vec3, projection: Mat4, screen_size: Vec2) -> Vec2 {
        // Calculate the View Matrix
        let view = Mat4::look_at_rh(self.position, self.get_target(), vec3(0.0, 1.0, 0.0));

        // Transform world position to clip space
        let clip_space_pos = projection * view * world_pos.extend(1.0);

        // Perspective divide (convert from 4D to 3D normalized device coordinates)
        let ndc = clip_space_pos.xyz() / clip_space_pos.w;

        // Convert NDC (-1 to 1) to screen coordinates
        let screen_x = (ndc.x + 1.0) * 0.5 * screen_size.x;
        let screen_y = (1.0 - (ndc.y + 1.0) * 0.5) * screen_size.y; // Flip Y

        vec2(screen_x, screen_y)
    }

    fn is_world_in_screen(&self, world_pos: Vec3, projection: Mat4) -> bool {
        // Compute the view matrix (camera transform)
        let view = Mat4::look_at_rh(self.position, self.get_target(), vec3(0.0, 1.0, 0.0));

        // Transform world position into clip space
        let clip_space_pos = projection * view * world_pos.extend(1.0);

        // Perspective divide (convert to Normalized Device Coordinates)
        let ndc = clip_space_pos.xyz() / clip_space_pos.w;

        // Check if the point is inside the [-1,1] range for X, Y, and Z
        ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0
    }

    fn update(&mut self, dont_move: bool) {
        // let (dx, dy) = mouse_delta(); // Get mouse movement
        let (dx, dy) = (mouse_position().0 - self.last_mouse_pos.x, mouse_position().1 - self.last_mouse_pos.y);
        self.last_mouse_pos = mouse_position().into();
        if !dont_move {
            self.yaw -= dx * -0.001; // Adjust sensitivity
            self.pitch += dy * -0.001;
        }

        // Clamp the pitch angle to prevent flipping
        self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Goofworld v0.1".to_owned(),
        fullscreen: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut camera = FreeLookCamera::new(vec3(0.0, 2.5, 5.0));

    let mut is_cursor_grabbed = true;
    
    let mut is_jumping = false;

    let mut camera_pos_change = vec3(0.0, 0.0, 0.0);

    let mut green_y = 1.0;
    let mut is_green_going_up = true;
    
    let planecat_texture = load_texture("assets/planecat.png").await.unwrap();
    let mut planecat_x = -10.0;
    let mut is_planecat_going_right = true;

    let mut is_sneaking;

    let holycrackers_texture = load_texture("assets/holycrackers.png").await.unwrap();

    let glungus_texture = load_texture("assets/glungus.png").await.unwrap();
    let mut glungus_position = vec3(-4.0, 1.0, 0.0);

    let mut rainbow_color = Color::new(0., 0., 0., 1.);
    let mut is_rainbow_r_up = true;
    let mut is_rainbow_g_up = true;
    let mut is_rainbow_b_up = true;

    // let table_model = load_model("assets/table.obj").await.unwrap();
    let table_model = load_model("assets/table.obj", (6., -0.5, -5.), Color::new(0., 1., 0.65, 0.5));

    loop {
        clear_background(BLACK);

        set_cursor_grab(is_cursor_grabbed);
        show_mouse(false);

        is_sneaking = is_key_down(KeyCode::LeftShift);

        camera.update(!is_cursor_grabbed);

        let cam3d = Camera3D {
            position: camera.position,
            target: camera.get_target(),
            up: vec3(0.0, 1.0, 0.0),
            fovy: deg2rad(100.0),
            ..Default::default()
        };

        if is_key_down(KeyCode::Escape) {
            break;
        }

        if is_key_down(KeyCode::W) {
            // camera.position += camera.get_target() - camera.position * 1.;
            // camera_pos_change = camera.get_target() - camera.position * 1.;
            let forward = (camera.get_target() - camera.position).normalize();
            camera_pos_change += vec3(forward.x, 0.0, forward.z) * 0.85;
        }

        if is_key_down(KeyCode::S) {
            // camera.position -= camera.get_target() - camera.position * 1.;
            // camera_pos_change = -(camera.get_target() - camera.position * 1.);
            let forward = (camera.get_target() - camera.position).normalize();
            camera_pos_change += -vec3(forward.x, 0.0, forward.z) * 0.85;
        }

        if is_key_down(KeyCode::A) {
            let forward = (camera.get_target() - camera.position).normalize();
            let right = forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
            // camera.position -= right * 1.;
            camera_pos_change += -right * 0.85;
        }

        if is_key_down(KeyCode::D) {
            let forward = (camera.get_target() - camera.position).normalize();
            let right = forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
            // camera.position += right * 1.;
            camera_pos_change += right * 0.85;
        }

        if is_key_down(KeyCode::Space) {
            if !is_jumping {
                camera_pos_change.y += 25.0;
                is_jumping = true;
            }
        }

        if is_key_down(KeyCode::Q) {
            camera_pos_change.x *= 1.25;
            camera_pos_change.z *= 1.25;
            if camera_pos_change.length() > 15.0 {
                camera_pos_change = camera_pos_change.normalize() * 15.0;
            }
        }

        if is_sneaking {
            camera_pos_change *= 0.75;
        }

        if is_key_pressed(KeyCode::LeftControl) {
            is_cursor_grabbed = !is_cursor_grabbed;
        }

        camera_pos_change.y -= 0.75; // Gravity

        // camera.position += camera_pos_change * 1. / 20.;
        camera.position.x += camera_pos_change.x / 30.;
        camera.position.y += camera_pos_change.y / 30.;
        camera.position.z += camera_pos_change.z / 20.;

        if camera.position.y < 2.5 {
            camera.position.y = 2.5;
            is_jumping = false;
        }

        camera.position.y -= if is_sneaking { 0.5 } else { 0.0 };
        
        if green_y >= 4.0 {
            is_green_going_up = false;
        } else if green_y <= 1.0 {
            is_green_going_up = true;
        }

        if is_green_going_up {
            green_y += 0.03;
        } else {
            green_y -= 0.03;
        }

        if planecat_x >= 10.0 {
            is_planecat_going_right = false;
        } else if planecat_x <= -10.0 {
            is_planecat_going_right = true;
        }

        if is_planecat_going_right {
            planecat_x += 0.025;
        } else {
            planecat_x -= 0.025;
        }

        // Cycle through the 16.7 million colors of RGB
        if is_rainbow_r_up {
            rainbow_color.r += 0.01;
            if rainbow_color.r >= 1.0 {
                is_rainbow_r_up = false;
            }
        } else {
            rainbow_color.r -= 0.01;
            if rainbow_color.r <= 0.0 {
                is_rainbow_r_up = true;
            }
        }
        if is_rainbow_g_up {
            rainbow_color.g += 0.02;
            if rainbow_color.g >= 1.0 {
                is_rainbow_g_up = false;
            }
        } else {
            rainbow_color.g -= 0.02;
            if rainbow_color.g <= 0.0 {
                is_rainbow_g_up = true;
            }
        }
        if is_rainbow_b_up {
            rainbow_color.b += 0.03;
            if rainbow_color.b >= 1.0 {
                is_rainbow_b_up = false;
            }
        } else {
            rainbow_color.b -= 0.03;
            if rainbow_color.b <= 0.0 {
                is_rainbow_b_up = true;
            }
        }
        
        camera_pos_change.x *= 0.75;
        camera_pos_change.z *= 0.75;
        camera_pos_change.y *= 0.9;

        if camera_pos_change.x.abs() < 0.01 {
            camera_pos_change.x = 0.0;
        }
        if camera_pos_change.z.abs() < 0.01 {
            camera_pos_change.z = 0.0;
        }
        if camera_pos_change.y.abs() < 0.01 {
            camera_pos_change.y = 0.0;
        }

        // Make glungus come closer to the camera
        if glungus_position.x < camera.position.x {
            glungus_position.x += (glungus_position.x - camera.position.x).abs() / 100.;
        } else {
            glungus_position.x -= (glungus_position.x - camera.position.x).abs() / 100.;
        }
        if glungus_position.z < camera.position.z {
            glungus_position.z += (glungus_position.z - camera.position.z).abs() / 100.;
        } else {
            glungus_position.z -= (glungus_position.z - camera.position.z).abs() / 100.;
        }

        set_camera(&cam3d);

        draw_grid(200, 1., LIGHTGRAY, GRAY);
        
        // Draw lines to show the axes
        draw_line_3d(vec3(0.0, 10.0, 0.0), vec3(10.0, 10.0, 0.0), RED); // X
        draw_line_3d(vec3(0.0, 10.0, 0.0), vec3(0.0, 20.0, 0.0), GREEN); // Y
        draw_line_3d(vec3(0.0, 10.0, 0.0), vec3(0.0, 10.0, 10.0), BLUE); // Z

        draw_cube(vec3(-2.0, 1.0, 1.0), vec3(1.0, 2.0, 1.0), None, RED);
        draw_cube(vec3(0.0, green_y, -0.5), vec3(2.0, 2.0, 2.0), None, GREEN);
        draw_cube(vec3(2.0, 1.0, -1.0), vec3(1.0, 2.0, 1.0), None, BLUE);
        draw_cube(vec3(4.0, 1.0, 2.0), vec3(2.0, 2.0, 1.0), Some(&holycrackers_texture), WHITE);

        draw_cube(vec3(planecat_x, 5.0, 0.0), vec3(2.0, 1.0, 1.0), Some(&planecat_texture), WHITE);

        for mesh in table_model.iter() {
            draw_mesh(mesh);
        }

        set_default_camera();

        // goofy little 3D text
        let projection = Mat4::perspective_rh_gl(100f32.to_radians(), screen_width() / screen_height(), 0.1, 100.0);

        if camera.is_world_in_screen(vec3(0.0, 10.0, 0.0), projection) {
            draw_text_3d(&camera, format!("{} GOOFWORLD! {}", random_string(1), random_string(1)).as_str(), vec3(0.0, 10.0, 0.0), projection, 64., rainbow_color);
        }

        // Draw glungus in 2D space, why? Because we can scale him based on distance from camera
        // and its too funny
        let scale = (1.0 / (camera.position - glungus_position).length()).clamp(0.001, 5.0);

        // Find the x and y position of the glungus on the screen, based on the 3D position
        let screen_pos = camera.world_to_screen(glungus_position, projection, vec2(screen_width(), screen_height()));
        
        if camera.is_world_in_screen(glungus_position, projection) {
            draw_texture_ex(&glungus_texture, screen_pos.x - (glungus_texture.width() * scale) / 2., screen_pos.y - (glungus_texture.height() * scale) / 2., WHITE, DrawTextureParams {
                dest_size: Some(vec2(800.0 * scale, 800.0 * scale)),
                ..Default::default()
            });
        }
        
        if is_cursor_grabbed {
            let screen_center = vec2(screen_width() / 2.0, screen_height() / 2.0);
            draw_line(screen_center.x - 10.0, screen_center.y, screen_center.x + 10.0, screen_center.y, 2.0, WHITE);
            draw_line(screen_center.x, screen_center.y - 10.0, screen_center.x, screen_center.y + 10.0, 2.0, WHITE);
        }

        draw_text(format!("Move with WASD keys, CTRL to {} mouse, Shift to sneak, Q to sprint and ESC to leave", if is_cursor_grabbed { "release" } else { "lock" } ).to_uppercase().as_str(), 20.0, 40.0, 32.0, WHITE);
        draw_text(format!("Camera position: {:?}", camera.position).to_uppercase().as_str(), 20.0, 80.0, 48.0, WHITE);
        
        let text_height = measure_text("A", None, 0, 80.0).height;
        draw_text(format!("FPS: {}", get_fps()).to_uppercase().as_str(), 20.0, screen_height() - text_height - 20.0, 80.0, WHITE);

        if !is_cursor_grabbed {
            let mouse_pos = mouse_position();
            draw_line(mouse_pos.0 - 10.0, mouse_pos.1, mouse_pos.0 + 10.0, mouse_pos.1, 2.0, WHITE);
            draw_line(mouse_pos.0, mouse_pos.1 - 10.0, mouse_pos.0, mouse_pos.1 + 10.0, 2.0, WHITE);
        }
        
        next_frame().await;
    }
}

