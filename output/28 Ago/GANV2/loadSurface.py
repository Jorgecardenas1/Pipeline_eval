# ----------------------------------------------
# Script Recorded by Ansys Electronics Desktop Version 2021.2.0
# 11:24:19  Aug 29, 2024
# ----------------------------------------------
import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.SetActiveProject("model")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.ImportDXF(
	[
		"NAME:options",
		"FileName:="		, "C:/Users/jorge/Documents/Projects Jorge C/DRUIDA PROJECT/POC/metasurface_AI_pipeline_V1/Pipeline_eval/output/28 Ago/GANV2/c2eb73a2/processed/output.dxf",
		"Scale:="		, 1E-06,
		"AutoDetectClosed:="	, True,
		"SelfStitch:="		, True,
		"DefeatureGeometry:="	, False,
		"DefeatureDistance:="	, 0,
		"RoundCoordinates:="	, True,
		"RoundNumDigits:="	, 4,
		"WritePolyWithWidthAsFilledPoly:=", True,
		"ImportMethod:="	, 1,
		"2DSheetBodies:="	, True,
		[
			"NAME:LayerInfo",
			[
				"NAME:0",
				"source:="		, "0",
				"display_source:="	, "0",
				"import:="		, False,
				"dest:="		, "0",
				"dest_selected:="	, False,
				"layer_type:="		, "signal"
			],
			[
				"NAME:Defpoints",
				"source:="		, "Defpoints",
				"display_source:="	, "Defpoints",
				"import:="		, False,
				"dest:="		, "Defpoints",
				"dest_selected:="	, False,
				"layer_type:="		, "signal"
			],
			[
				"NAME:conductor",
				"source:="		, "conductor",
				"display_source:="	, "conductor",
				"import:="		, True,
				"dest:="		, "conductor",
				"dest_selected:="	, False,
				"layer_type:="		, "signal"
			],
			[
				"NAME:dielectric",
				"source:="		, "dielectric",
				"display_source:="	, "dielectric",
				"import:="		, False,
				"dest:="		, "dielectric",
				"dest_selected:="	, False,
				"layer_type:="		, "signal"
			],
			[
				"NAME:substrate",
				"source:="		, "substrate",
				"display_source:="	, "substrate",
				"import:="		, False,
				"dest:="		, "substrate",
				"dest_selected:="	, False,
				"layer_type:="		, "signal"
			]
		],
		[
			"NAME:TechFileLayers",
			"layer:="		, [				"SrcName:="		, "conductor",				"DestName:="		, "conductor",				"Thickness:="		, 3.5E-05,				"Elevation:="		, 0.000508,				"Color:="		, "red"],
			"layer:="		, [				"SrcName:="		, "dielectric",				"DestName:="		, "dielectric",				"Thickness:="		, 0.000508,				"Elevation:="		, 0.000508,				"Color:="		, "green"],
			"layer:="		, [				"SrcName:="		, "substrate",				"DestName:="		, "substrate",				"Thickness:="		, 0.000508,				"Elevation:="		, 0,				"Color:="		, "blue"]
		]
	])
oEditor.Delete(
	[
		"NAME:Selections",
		"Selections:="		, "conductor_2"
	])
oEditor.ThickenSheet(
	[
		"NAME:Selections",
		"Selections:="		, "conductor_1",
		"NewPartsModelFlag:="	, "Model"
	], 
	[
		"NAME:SheetThickenParameters",
		"Thickness:="		, "0.035mm",
		"BothSides:="		, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"conductor_1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Material",
					"Value:="		, "\"copper\""
				]
			]
		]
	])
oModule = oDesign.GetModule("MeshSetup")
oModule.AssignModelResolutionOp(
	[
		"NAME:ModelResolution1",
		"Objects:="		, ["conductor_1"],
		"UseAutoLength:="	, False,
		"DefeatureLength:="	, "0.08mm"
	])
oDesign.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:LocalVariableTab",
			[
				"NAME:PropServers", 
				"LocalVariables"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:H",
					"Value:="		, "0.757mm"
				]
			]
		]
	])
oDesign.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:LocalVariableTab",
			[
				"NAME:PropServers", 
				"LocalVariables"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:H",
					"Value:="		, "0.508mm"
				]
			]
		]
	])
oProject.Save()
oDesign.AnalyzeAll()
oModule = oDesign.GetModule("ReportSetup")
oModule.ExportToFile("Absorption", "C:/Users/jorge/Documents/Projects Jorge C/DRUIDA PROJECT/POC/metasurface_AI_pipeline_V1/Pipeline_eval/output/28 Ago/GANV2/c2eb73a2/Absorption.csv")
