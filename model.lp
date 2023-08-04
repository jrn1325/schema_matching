\ Model quadratic
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  T_t_name.firstname-----S_s_name.firstname
   + T_t_name.lastname-----S_s_name.firstname + T_t_age-----S_s_age
   + T_t_gender-----S_s_gender
Subject To
 Matching_nested_target_implies_matching_root_target.:
   T_t_name.firstname-----S_sROOT_s_name
   - T_tROOT_t_name-----S_sROOT_s_name <= 0
 Matching_nested_source_implies_matching_root_source.:
   - T_tROOT_t_name-----S_sROOT_s_name
   + T_tROOT_t_name-----S_s_name.firstname <= 0
 Matching_nested_target_implies_matching_root_target.:
   - T_tROOT_t_name-----S_sROOT_s_name
   + T_t_name.lastname-----S_sROOT_s_name <= 0
 Matching_nested_source_implies_matching_root_source.:
   - T_tROOT_t_name-----S_sROOT_s_name
   + T_tROOT_t_name-----S_s_name.firstname <= 0
 Matching_nested_target_implies_matching_root_target.: T_t_age-----S_sROOT_
   - T_tROOT_-----S_sROOT_ <= 0
 Matching_nested_source_implies_matching_root_source.:
   - T_tROOT_-----S_sROOT_ + T_tROOT_-----S_s_age <= 0
 Matching_nested_target_implies_matching_root_target.:
   - T_tROOT_-----S_sROOT_ + T_t_gender-----S_sROOT_ <= 0
 Matching_nested_source_implies_matching_root_source.:
   - T_tROOT_-----S_sROOT_ + T_tROOT_-----S_s_gender <= 0
 Each_source_key_can_be_matched_to_at_most_one_target_key.:
   T_t_gender-----S_s_gender <= 1
Bounds
Binaries
 T_t_name.firstname-----S_s_name.firstname
 T_t_name.firstname-----S_sROOT_s_name T_tROOT_t_name-----S_sROOT_s_name
 T_tROOT_t_name-----S_s_name.firstname
 T_t_name.lastname-----S_s_name.firstname
 T_t_name.lastname-----S_sROOT_s_name T_tROOT_t_name-----S_s_name.firstname
 T_t_age-----S_s_age T_t_age-----S_sROOT_ T_tROOT_-----S_sROOT_
 T_tROOT_-----S_s_age T_t_gender-----S_s_gender T_t_gender-----S_sROOT_
 T_tROOT_-----S_s_gender
End
