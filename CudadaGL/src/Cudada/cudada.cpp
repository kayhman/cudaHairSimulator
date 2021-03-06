#include "hairlib.h"
#include <windows.h>		
#include <gl\gl.h>			
#include <gl\glu.h>			
	
#include "objLoader.h"

HDC			hDC=NULL;		
HGLRC		hRC=NULL;		
HWND		hWnd=NULL;		
HINSTANCE	hInstance;		

bool	keys[256];			
bool	active=TRUE;		
bool	fullscreen=TRUE;	

GLfloat	rtri;				
GLfloat	rquad;				

LRESULT	CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);	

GLvoid ReSizeGLScene(GLsizei width, GLsizei height)		
{
	if (height==0)										
	{
		height=1;										
	}

	glViewport(0,0,width,height);						

	glMatrixMode(GL_PROJECTION);						
	glLoadIdentity();									

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							
	glLoadIdentity();									
}

int InitGL(GLvoid)										
{
   
   //GLfloat mat_shininess[] = { 500.0 };
   GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_SMOOTH);


   //glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
   glLightfv(GL_LIGHT0, GL_POSITION, light_position);

   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);

   //glEnable(GL_NORMALIZE);

	return TRUE;										
}

int DrawGLScene(GLvoid)									
{		
	const float hxy = 0.01;
	const float hz = 0.15;
	const float dt = 1e-2;
	const int hairLenght = 30;
	const int gridX = 128;
	const int gridY = 128;

	static ObjLoader loader("cudada.obj");


	static HairSimulation simu(0., 0., 0., 3.0, gridX, gridY, hairLenght, hxy, hz);
	static bool init = false;
	if(!init)
		simu.initHair();
	init = true;




	simu.integrate(dt);
	float maxZ = hairLenght * hz;

	std::vector<float> X = simu.getMassPositionX();
	std::vector<float> Y = simu.getMassPositionY();
	std::vector<float> Z = simu.getMassPositionZ();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	glLoadIdentity();

	glTranslatef(0.f,0.f,-32.0f);			
	glRotatef(rtri,0.0f,1.0f,0.0f);
	//glRotatef(45,0.0f,1.0f,0.0f);
	
	//{
	//	GLfloat mat_ambient[] = { 1.0, 1.0, 0.0, 1.0 };
	//	GLfloat mat_specular[] = { 0.2, 0.15, 0.15, 1.0 };

	//	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	//	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	//}

	for(int i = 0 ; i < gridX ; ++i)
		for(int j = 0 ; j < gridY ; ++j)
		{
			if(i%8 == 0)		
				if(j % 8 == 0)
				{
					glBegin(GL_LINE_STRIP);
					for(int z = 0 ; z < hairLenght ; ++z)
					{
						int idx = z * gridX * gridY + j * gridX + i;
						glColor3f(0.9, 0.9, 0.7);
						glVertex3f( X[idx], Y[idx], Z[idx]);
						//glNormal3f( 0., 0., 1.0);
					}
					glEnd();
				}
		}


	std::vector<float>& vertices = loader.getVertices();
	std::vector<float>& normals = loader. getVerticesNormal();
	std::vector<float>& triangleNormals = loader. getTrianglesNormal();
	std::vector<int>& triangles = loader. getTriangles();

	
	glRotatef(90,1.0f,0.0f,0.0f);
	//glTranslatef(0.f,0.f,-32.0f);	
	//glRotatef(90,0.0f,1.0f,0.0f);
	//glRotatef(90,1.0f,0.0f,0.0f);
	//glRotatef(180,0.0f,1.0f,0.0f);
	//glRotatef(-90,1.0f,0.0f,0.0f);
	//glRotatef(45,0.0f,0.0f,1.0f);

	{
		GLfloat mat_ambient[] = { 1.0, 0.0, 0.0, 1.0 };
		GLfloat mat_specular[] = { 0.2, 0.15, 0.15, 1.0 };
		GLfloat mat_emissive[] = { 1.0, 0.0, 0.0, 1.0 };

		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
		glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
		glMaterialfv(GL_FRONT, GL_EMISSION, mat_emissive);
	}
	glBegin(GL_TRIANGLES);
	for(int i = 0 ; i < triangles.size()/3 ; i++)
	{
		float Ax = vertices[3*triangles[3*i] + 0];
		float Ay = vertices[3*triangles[3*i] + 1];
		float Az = vertices[3*triangles[3*i] + 2];

		float Bx = vertices[3*triangles[3*i+1] + 0];
		float By = vertices[3*triangles[3*i+1] + 1];
		float Bz = vertices[3*triangles[3*i+1] + 2];


		float Cx = vertices[3*triangles[3*i+2] + 0];
		float Cy = vertices[3*triangles[3*i+2] + 1];
		float Cz = vertices[3*triangles[3*i+2] + 2];

		float Anx = normals[3*triangles[3*i] + 0];
		float Any = normals[3*triangles[3*i] + 1];
		float Anz = normals[3*triangles[3*i] + 2];

		float Bnx = normals[3*triangles[3*i+1] + 0];
		float Bny = normals[3*triangles[3*i+1] + 1];
		float Bnz = normals[3*triangles[3*i+1] + 2];

		float Cnx = normals[3*triangles[3*i+2] + 0];
		float Cny = normals[3*triangles[3*i+2] + 1];
		float Cnz = normals[3*triangles[3*i+2] + 2];

		glNormal3f( Anx, Any, Anz);
		glVertex3f( Ax, Ay, Az);
		glColor3f(1.0, 0.0, 0.0);

		glNormal3f( Bnx, Bny, Bnz);
		glVertex3f( Bx, By, Bz);
		glColor3f(1.0, 0.0, 0.0);

		glNormal3f( Cnx, Cny, Cnz);
		glVertex3f( Cx, Cy, Cz);
		glColor3f(1.0, 0.0, 0.0);


	}
	glEnd();
	rtri+=0.2f;									
	return TRUE;										
}

GLvoid KillGLWindow(GLvoid)
{
	if (fullscreen)										
	{
		ChangeDisplaySettings(NULL,0);					
		ShowCursor(TRUE);								
	}

	if (hRC)											
	{
		if (!wglMakeCurrent(NULL,NULL))					
		{
			MessageBox(NULL,"Release Of DC And RC Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		}

		if (!wglDeleteContext(hRC))						// Are We Able To Delete The RC?
		{
			MessageBox(NULL,"Release Rendering Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		}
		hRC=NULL;										// Set RC To NULL
	}

	if (hDC && !ReleaseDC(hWnd,hDC))					// Are We Able To Release The DC
	{
		MessageBox(NULL,"Release Device Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		hDC=NULL;										// Set DC To NULL
	}

	if (hWnd && !DestroyWindow(hWnd))					// Are We Able To Destroy The Window?
	{
		MessageBox(NULL,"Could Not Release hWnd.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		hWnd=NULL;										// Set hWnd To NULL
	}

	if (!UnregisterClass("OpenGL",hInstance))			// Are We Able To Unregister Class
	{
		MessageBox(NULL,"Could Not Unregister Class.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		hInstance=NULL;									// Set hInstance To NULL
	}
}

 
BOOL CreateGLWindow(char* title, int width, int height, int bits, bool fullscreenflag)
{
	GLuint		PixelFormat;			
	WNDCLASS	wc;						
	DWORD		dwExStyle;				
	DWORD		dwStyle;				
	RECT		WindowRect;				
	WindowRect.left=(long)0;			
	WindowRect.right=(long)width;		
	WindowRect.top=(long)0;				
	WindowRect.bottom=(long)height;		

	fullscreen=fullscreenflag;			

	hInstance			= GetModuleHandle(NULL);				
	wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC;	
	wc.lpfnWndProc		= (WNDPROC) WndProc;					
	wc.cbClsExtra		= 0;									
	wc.cbWndExtra		= 0;									
	wc.hInstance		= hInstance;							
	wc.hIcon			= LoadIcon(NULL, IDI_WINLOGO);			
	wc.hCursor			= LoadCursor(NULL, IDC_ARROW);			
	wc.hbrBackground	= NULL;									
	wc.lpszMenuName		= NULL;									
	wc.lpszClassName	= "OpenGL";								

	if (!RegisterClass(&wc))									
	{
		MessageBox(NULL,"Failed To Register The Window Class.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;											
	}
	
	if (fullscreen)												
	{
		DEVMODE dmScreenSettings;								
		memset(&dmScreenSettings,0,sizeof(dmScreenSettings));	
		dmScreenSettings.dmSize=sizeof(dmScreenSettings);		
		dmScreenSettings.dmPelsWidth	= width;				
		dmScreenSettings.dmPelsHeight	= height;				
		dmScreenSettings.dmBitsPerPel	= bits;					
		dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

		// Try To Set Selected Mode And Get Results.  NOTE: CDS_FULLSCREEN Gets Rid Of Start Bar.
		if (ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
		{
			// If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
			if (MessageBox(NULL,"The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?","NeHe GL",MB_YESNO|MB_ICONEXCLAMATION)==IDYES)
			{
				fullscreen=FALSE;		// Windowed Mode Selected.  Fullscreen = FALSE
			}
			else
			{
				// Pop Up A Message Box Letting User Know The Program Is Closing.
				MessageBox(NULL,"Program Will Now Close.","ERROR",MB_OK|MB_ICONSTOP);
				return FALSE;									// Return FALSE
			}
		}
	}

	if (fullscreen)												
	{
		dwExStyle=WS_EX_APPWINDOW;								
		dwStyle=WS_POPUP;										
		ShowCursor(FALSE);										
	}
	else
	{
		dwExStyle=WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;			
		dwStyle=WS_OVERLAPPEDWINDOW;							
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);		

	// Create The Window
	if (!(hWnd=CreateWindowEx(	dwExStyle,							
								"OpenGL",							
								title,								
								dwStyle |							
								WS_CLIPSIBLINGS |					
								WS_CLIPCHILDREN,					
								0, 0,								
								WindowRect.right-WindowRect.left,	
								WindowRect.bottom-WindowRect.top,	
								NULL,								
								NULL,								
								hInstance,							
								NULL)))								
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Window Creation Error.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	static	PIXELFORMATDESCRIPTOR pfd=				
	{
		sizeof(PIXELFORMATDESCRIPTOR),				
		1,											
		PFD_DRAW_TO_WINDOW |						
		PFD_SUPPORT_OPENGL |						
		PFD_DOUBLEBUFFER,							
		PFD_TYPE_RGBA,								
		bits,										
		0, 0, 0, 0, 0, 0,							
		0,											
		0,											
		0,											
		0, 0, 0, 0,									
		16,											
		0,											
		0,											
		PFD_MAIN_PLANE,								
		0,											
		0, 0, 0										
	};
	
	if (!(hDC=GetDC(hWnd)))							
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Can't Create A GL Device Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if (!(PixelFormat=ChoosePixelFormat(hDC,&pfd)))	// Did Windows Find A Matching Pixel Format?
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Can't Find A Suitable PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if(!SetPixelFormat(hDC,PixelFormat,&pfd))		// Are We Able To Set The Pixel Format?
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Can't Set The PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if (!(hRC=wglCreateContext(hDC)))				// Are We Able To Get A Rendering Context?
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Can't Create A GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if(!wglMakeCurrent(hDC,hRC))					// Try To Activate The Rendering Context
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Can't Activate The GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	ShowWindow(hWnd,SW_SHOW);						
	SetForegroundWindow(hWnd);						
	SetFocus(hWnd);									
	ReSizeGLScene(width, height);					

	if (!InitGL())									// Initialize Our Newly Created GL Window
	{
		KillGLWindow();								// Reset The Display
		MessageBox(NULL,"Initialization Failed.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	return TRUE;									// Success
}

LRESULT CALLBACK WndProc(	HWND	hWnd,			
							UINT	uMsg,			
							WPARAM	wParam,			
							LPARAM	lParam)			
{
	switch (uMsg)									
	{
		case WM_ACTIVATE:							
		{
			// LoWord Can Be WA_INACTIVE, WA_ACTIVE, WA_CLICKACTIVE,
			// The High-Order Word Specifies The Minimized State Of The Window Being Activated Or Deactivated.
			// A NonZero Value Indicates The Window Is Minimized.
			if ((LOWORD(wParam) != WA_INACTIVE) && !((BOOL)HIWORD(wParam)))
				active=TRUE;						
			else
				active=FALSE;						

			return 0;								
		}

		case WM_SYSCOMMAND:							
		{
			switch (wParam)							
			{
				case SC_SCREENSAVE:					
				case SC_MONITORPOWER:				
				return 0;							
			}
			break;									
		}

		case WM_CLOSE:								
		{
			PostQuitMessage(0);						
			return 0;								
		}

		case WM_KEYDOWN:							
		{
			keys[wParam] = TRUE;					
			return 0;								
		}

		case WM_KEYUP:								
		{
			keys[wParam] = FALSE;					
			return 0;								
		}

		case WM_SIZE:								
		{
			ReSizeGLScene(LOWORD(lParam),HIWORD(lParam));  // LoWord=Width, HiWord=Height
			return 0;								// Jump Back
		}
	}

	// Pass All Unhandled Messages To DefWindowProc
	return DefWindowProc(hWnd,uMsg,wParam,lParam);
}

int WINAPI WinMain(	HINSTANCE	hInstance,			
					HINSTANCE	hPrevInstance,		
					LPSTR		lpCmdLine,			
					int			nCmdShow)			
{
	MSG		msg;									
	BOOL	done=FALSE;	




	// Ask The User Which Screen Mode They Prefer
	if (MessageBox(NULL,"Would You Like To Run In Fullscreen Mode?", "Start FullScreen?",MB_YESNO|MB_ICONQUESTION)==IDNO)
	{
		fullscreen=FALSE;							// Windowed Mode
	}

	// Create Our OpenGL Window
	if (!CreateGLWindow("NeHe's Solid Object Tutorial",640,480,16,fullscreen))
	{
		return 0;									// Quit If Window Was Not Created
	}

	while(!done)									
	{
		if (PeekMessage(&msg,NULL,0,0,PM_REMOVE))	
		{
			if (msg.message==WM_QUIT)				
			{
				done=TRUE;							
			}
			else									
			{
				TranslateMessage(&msg);				
				DispatchMessage(&msg);				
			}
		}
		else										// If There Are No Messages
		{
			// Draw The Scene.  Watch For ESC Key And Quit Messages From DrawGLScene()
			if ((active && !DrawGLScene()) || keys[VK_ESCAPE])	// Active?  Was There A Quit Received?
			{
				done=TRUE;							
			}
			else									
			{
				SwapBuffers(hDC);					
			}

			if (keys[VK_F1])						
			{
				keys[VK_F1]=FALSE;					
				KillGLWindow();						
				fullscreen=!fullscreen;				
				// Recreate Our OpenGL Window
				if (!CreateGLWindow("NeHe's Solid Object Tutorial",640,480,16,fullscreen))
				{
					return 0;						// Quit If Window Was Not Created
				}
			}
		}
	}

	// Shutdown
	KillGLWindow();									// Kill The Window
	return (msg.wParam);							// Exit The Program
}
