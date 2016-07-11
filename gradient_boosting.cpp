#include "gradient_boosting.h"

#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

int max_feature_label(string line)
{
	int start = 0;
	int fid;
	int max_fid = 0;
	int len = line.length();
 
	for(int i = 0; i < len; i++) {
		if(line[i] == ' ') {
			start = i+1;
		}
		else if(line[i] == ':') {
			if(sscanf(line.substr(start, i - start).c_str(), "%d", &fid) == 1) {
				if(max_fid < fid) {
					max_fid = fid;
				}
			}	
		}
	}
 
	return max_fid;
}
 

int splitline(string line, string items[], int items_num, const char separator)
{
	if(items == NULL || items_num <= 0) {
		return -1;
 	}
/*	printf("%s\n", line.c_str());	*/
	int n = line.length();
 	int j = 0;
 	int start = 0;
 	for(int i = 0; i < n; i++) {
	/*	LOG_ERROR_(line[i]);*/
		if(line[i] == separator) {
			if(j < items_num && start < n) {
				items[j++] = line.substr(start, i-start);
				start = ++i;
			}
		}
 	}

 	if(j < items_num && start < n) {
		items[j++] = line.substr(start, n-start);
 	}

 	return j;
}

int gbdt_tree_node_split( 
		gbdt_info_t gbdt_inf, 
		bufset* data_set, 
		double *x_fea_value, 
		double *y_score,
	    nodeinfo ninf, 
		int* index, 
		splitinfo* spinf)
{
	if(data_set == NULL || x_fea_value == NULL || y_score == NULL || index == NULL || spinf == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}
 	spinf->bestsplit = -1;
 	spinf->bestid = -1;

 	for (int i=0; i < gbdt_inf.fea_num; ++i)
 	{
		data_set->fea_pool[i] = i;
 	}
 	int last = gbdt_inf.fea_num - 1;

 	double critmax = 0.0;

 	for (int i = 0; i < gbdt_inf.rand_fea_num; ++i) 
 	{
		//debug
		//int select = last;

		int select = rand() % (last+1);
 		int fid = data_set->fea_pool[select]; // fid = ¿¿¿id 0 - max_id
 		data_set->fea_pool[select] = data_set->fea_pool[last]; 
 		data_set->fea_pool[last] = fid;
 		last--;

		for (int j = ninf.index_b; j <= ninf.index_e; j++){
			data_set->fvalue_list[j] = x_fea_value[index[j]* gbdt_inf.fea_num + fid]; 

 		 	data_set->fv[j] = data_set->fvalue_list[j]; 

 		 	data_set->y_list[j] = y_score[index[j]]; 
 		}
 		
 		for (int j = 0; j < gbdt_inf.sample_num; ++j) {
			data_set->order_i[j] = j;
 		}
 		
 		R_qsort_I(data_set->fv, data_set->order_i, ninf.index_b+1 , ninf.index_e+1);
 	
 	 // debug
 	 //for (int s = 0; s<300; s++)
 	 //{
 	 // printf("%d==%lf_%d\n", s, data_set->fv[s],data_set->order_i[s]);
 	 //}
 	
		if (data_set->fv[ninf.index_b] >= data_set->fv[ninf.index_e]) {
			continue; // 
 	 	}
 	
 	
		double left_sum = 0.0;
 	 	int left_num = 0;
 	 	double right_sum = ninf.nodesum;
 	 	int right_num = ninf.nodenum;
 	 	double d = 0.0;
 	 	double crit = 0.0;
 	 	double tmpsplit = 0;
 	 	double critvar = 0;
 	
		for (int j=ninf.index_b; j< ninf.index_e; j++)
 	 	{
 	 	// d = y_result_score[data_set->order_i[j]];
			d = data_set->y_list[data_set->order_i[j]];
 	 		left_sum  += d;
 	 		right_sum -= d;
 	 		left_num++;
 	 		right_num--;
			if (data_set->fv[j] < data_set->fv[j+1]) {
				crit = (left_sum * left_sum / left_num) + (right_sum * right_sum / right_num) - ninf.critparent;
 	 			if (crit > critvar) {
					// ¿¿¿¿¿¿ feature value
					tmpsplit = (data_set->fv[j] + data_set->fv[j+1]) / 2.0; 
					critvar = crit;
				}
 	 		}
		}
 	
		// ¿¿¿¿feature¿¿¿ critvar > cirtmax, ¿¿¿¿
		if (critvar > critmax) {
			spinf->bestsplit = tmpsplit; // split feature vale 
 	 		spinf->bestid = fid;         // split feature id
 	 		critmax = critvar;           // split crit vaule
 	 	}
	}
 	
	// ¿¿bestid ¿¿node¿¿¿index
 	if( spinf->bestid >= 0) 
 	{
		int nl = ninf.index_b;
		for (int j= ninf.index_b; j<= ninf.index_e; j++)
		{	
			if (x_fea_value[index[j]* gbdt_inf.fea_num + spinf->bestid] <= spinf->bestsplit)
			{
				data_set->order_i[nl] = index[j]; //update data->set
				nl++;
			}
		}
		int nr = nl;
		for (int j= ninf.index_b; j<= ninf.index_e; j++)
		{
			if (x_fea_value[index[j]* gbdt_inf.fea_num + spinf->bestid] > spinf->bestsplit)
			{
				data_set->order_i[nr] = index[j];
				nr++;
			}
		}
		for (int j= ninf.index_b; j<= ninf.index_e; j++)
		{
			index[j] = data_set->order_i[j];
		}
 	
		spinf->pivot = nl;
 	
		return 0;
	}
 	else
 	{
		return 1;
 	}
}

int gbdt_single_tree_estimation(
		double *x_fea_value, 
		double *y_gradient, 
		gbdt_info_t gbdt_inf, 
		bufset* data_set, 
		int* index, 
		gbdt_tree_t* gbdt_single_tree, 
		int nrnodes )
{
	if(x_fea_value == NULL || y_gradient == NULL || data_set == NULL || index == NULL || gbdt_single_tree == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}
	
	splitinfo* spinf = (splitinfo*) malloc( sizeof(splitinfo));
 	if(spinf == NULL) {
		LOG_ERROR_("Failed to allocate memory.");
		return -1;
 	}
 	spinf->bestid = -1;

	for (int i = 0; i < gbdt_inf.sample_num; ++i) 	index[i] = i;
 	

	int ncur = 0;  
 	gbdt_single_tree->nodestatus[0] = GBDT_TOSPLIT;
 	gbdt_single_tree->ndstart[0]	= 0;
 	gbdt_single_tree->ndcount[0]	= gbdt_inf.sample_num;
 	gbdt_single_tree->depth[0]		= 0;

	/* compute mean and sum of squares for Y */
 	double avg = 0.0;
 	for (int i = 0; i < gbdt_inf.sample_num; ++i) 
		avg = (i * avg + y_gradient[index[i]]) / (i + 1); 
 	
 	gbdt_single_tree->ndavg[0] = avg;
	if (gbdt_single_tree->ndcount[0] <= gbdt_inf.gbdt_min_node_size) 
 	{
		gbdt_single_tree->nodestatus[0] = GBDT_TERMINAL;
 		gbdt_single_tree->lson[0]		= 0; // debug temp
 		gbdt_single_tree->rson[0]		= 0;
 		gbdt_single_tree->splitid[0]	= 0;
 		gbdt_single_tree->splitvalue[0] = 0.0;

 		gbdt_single_tree->nodesize		= 1;
 		free(spinf);
 		return 0;
 	}

	/* start main loop */
 	for (int k = 0; k < nrnodes - 2; ++k) {
		if (k > ncur || ncur >= nrnodes - 2) {
			break;
		}

		/* skip if the node is not to be split */ 
 		if (gbdt_single_tree->nodestatus[k] != GBDT_TOSPLIT) {
			continue;
 		}

		/* initialize for next call to findbestsplit */
 		nodeinfo ninf;
 		ninf.index_b = gbdt_single_tree->ndstart[k];
 		ninf.index_e = gbdt_single_tree->ndstart[k] + gbdt_single_tree->ndcount[k] - 1;
 		ninf.nodenum = gbdt_single_tree->ndcount[k];
 		ninf.nodesum = gbdt_single_tree->ndcount[k] * gbdt_single_tree->ndavg[k];
 		ninf.critparent = (ninf.nodesum * ninf.nodesum) / ninf.nodenum;

 		int jstat;

		jstat = gbdt_tree_node_split(gbdt_inf, data_set, x_fea_value, y_gradient, ninf, index, spinf);

 		if (jstat == 1) // 
 		{
			/* Node is terminal: Mark it as such and move on to the next. */
 			gbdt_single_tree->nodestatus[k] = GBDT_TERMINAL;
 			continue;
 		}
 		if(jstat == -1)
 		{
			free(spinf);
			return -1;
 		}

 		gbdt_single_tree->splitid[k] = spinf->bestid;
 		gbdt_single_tree->splitvalue[k] = spinf->bestsplit;
 		gbdt_single_tree->nodestatus[k] = GBDT_INTERIOR;

		/* leftnode no.= ncur+1, rightnode no. = ncur+2. */
 		gbdt_single_tree->ndstart[ncur + 1] = ninf.index_b;
 		gbdt_single_tree->ndstart[ncur + 2] = spinf->pivot;
 		gbdt_single_tree->ndcount[ncur + 1] = spinf->pivot - ninf.index_b; //
 		gbdt_single_tree->ndcount[ncur + 2] = ninf.index_e - spinf->pivot + 1;

 		gbdt_single_tree->depth[ncur + 1] = gbdt_single_tree->depth[k] + 1;
 		gbdt_single_tree->depth[ncur + 2] = gbdt_single_tree->depth[k] + 1;

 		/* compute mean and sum of squares for the left son node */
 		double avg = 0.0;
 		double d  = 0.0;
 		int m = 0;
 		for (int j = ninf.index_b; j < spinf->pivot; ++j) // mean
 		{
			d = y_gradient[index[j]];
			m = j - ninf.index_b;
			avg = (m * avg + d) / (m+1);
 		}

		double var = 0.0;
 		for (int j = ninf.index_b; j < spinf->pivot; ++j)
 		{
			var += (y_gradient[index[j]] - avg) * (y_gradient[index[j]] - avg);
 		}
 		var /= (spinf->pivot - ninf.index_b);

		gbdt_single_tree->ndavg[ncur+1] = avg;
 		gbdt_single_tree->nodestatus[ncur+1] = GBDT_TOSPLIT;
 		if (gbdt_single_tree->ndcount[ncur + 1] <= gbdt_inf.gbdt_min_node_size) 
 		{
			gbdt_single_tree->nodestatus[ncur + 1] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 1]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 1]       = 0;
 			gbdt_single_tree->splitid[ncur + 1]    = 0;
 			gbdt_single_tree->splitvalue[ncur + 1] = 0.0;
 		}

		if (gbdt_single_tree->depth[ncur + 1] >= gbdt_inf.gbdt_max_depth) 
 		{
			gbdt_single_tree->nodestatus[ncur + 1] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 1]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 1]       = 0;
 			gbdt_single_tree->splitid[ncur + 1]    = 0;
 			gbdt_single_tree->splitvalue[ncur + 1] = 0.0;
 		}

		// 
 		//if (var <= xxx)
 		//{
 		// gbdt_single_tree->nodestatus[ncur + 1] = GBDT_TERMINAL;
 		//}
		//

 		/* compute mean and sum of squares for the right daughter node */
 		avg = 0.0;
 		d   = 0.0;
 		m   = 0;
 		for (int j = spinf->pivot; j <= ninf.index_e; ++j) { 
			d   = y_gradient[index[j]];
 			m   = j - spinf->pivot;
 			avg = (m * avg + d) / (m + 1);
 		}
 		var = 0.0;
 		for (int j = spinf->pivot; j <= ninf.index_e; ++j) 
 		{
			var += (y_gradient[index[j]] - avg) * (y_gradient[index[j]] - avg);
 		}
 		var /= (ninf.index_e - spinf->pivot +1);

 		gbdt_single_tree->ndavg[ncur+2] = avg;
 		gbdt_single_tree->nodestatus[ncur+2] = GBDT_TOSPLIT;

 		if (gbdt_single_tree->ndcount[ncur + 2] <= gbdt_inf.gbdt_min_node_size) 
 		{
			gbdt_single_tree->nodestatus[ncur + 2] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 2]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 2]       = 0;
 			gbdt_single_tree->splitid[ncur + 2]    = 0;
 			gbdt_single_tree->splitvalue[ncur +2]  = 0.0;
 		}

 		if (gbdt_single_tree->depth[ncur + 2] >= gbdt_inf.gbdt_max_depth) 
 		{
			gbdt_single_tree->nodestatus[ncur + 2] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 2]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 2]       = 0;
 			gbdt_single_tree->splitid[ncur + 2]    = 0;
 			gbdt_single_tree->splitvalue[ncur +2]  = 0.0;
 		}

		//
		// if (var <= xxx)
 		// {
 		//    gbdt_single_tree->nodestatus[ncur + 2] = GBDT_TERMINAL;
 		// }
		//

 		gbdt_single_tree->lson[k] = ncur +1;
 		gbdt_single_tree->rson[k] = ncur +2;

 		ncur += 2;
	}

	gbdt_single_tree->nodesize = ncur+1;

	free(spinf);

	return 0;
}
  
int gbdt_regression_predict(gbdt_model_t* gbdt_model, double *x_test, double& ypredict)
{
	if(x_test == NULL || gbdt_model == NULL || gbdt_model->reg_forest == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}

 	ypredict = 0.0;
 	for (int i=0; i<gbdt_model->info.tree_num; i++)
 	{
		if(gbdt_model->reg_forest[i] != NULL)
 		{
			if(gbdt_tree_predict(x_test, gbdt_model->reg_forest[i], ypredict, gbdt_model->info.shrink) != 0)
			{
			return -1;
			}
 		}
 		else
			return -1;
 	}
 	return 0;
}
  
int gbdt_tree_predict(double *x_test, gbdt_tree_t *gbdt_single_tree, double& ypred, double shrink)
{
	if(x_test == NULL || gbdt_single_tree == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}
 	int k = 0;
 	while (gbdt_single_tree->nodestatus[k] != GBDT_TERMINAL) 
 	{ /* go down the tree */
		int m = gbdt_single_tree->splitid[k];
 		if (x_test[m] <= gbdt_single_tree->splitvalue[k])
 		{
			k = gbdt_single_tree->lson[k];
 		}
 		else
 		{
			k = gbdt_single_tree->rson[k];
 		}
 	}
 	ypred += shrink * gbdt_single_tree->ndavg[k];

 	return 0;
}
  
/*************************************************************************
* training function
*************************************************************************/

int print_usage (FILE* stream, char* program_name)
{
	fprintf (stream, "Usage: %s options [ inputfile ... ]\n", program_name);
 	fprintf (stream, " -h --help Êä³ö°ïÖúÐÅÏ¢.\n"
 	" -r --sample_feature_ratio Feature²ÉÑùÂÊ.\n"
 	" -t --tree_num GBDTÊ÷µÄÊýÄ¿.\n"
 	" -s --shrink Ñ§Ï°ÂÊ.\n"
 	" -n --min_node_size Ê÷µÄÉú³¤Í£Ö¹Ìõ¼þ:½Úµã¸²¸ÇµÄ×îÐ¡Êý¾ÝÁ¿.\n"
 	" -d --max_depth Ê÷µÄÉú³¤Í£Ö¹Ìõ¼þ:Ê÷µÄ×î´óÉî¶È.\n"
 	" -m --model_out Êä³öµÄÄ£ÐÍÎÄ¼þÃû.\n"
 	" -f --train_file ÑµÁ·Êý¾ÝÎÄ¼þ.\n"
 	);

 	return 0;
}
  
int init_info(gbdt_info_t& infbox)
{
	infbox.data_num = -1;
 	infbox.fea_num = -1;
 	infbox.gbdt_max_depth = -1;
 	infbox.gbdt_min_node_size = -1;
 	strcpy(infbox.model_filename, "");
 	strcpy(infbox.train_filename, "");
 	infbox.rand_fea_num = -1;
 	infbox.sample_num = -1;
 	infbox.shrink = -1;
 	infbox.tree_num = -1;

 	return 0;
}
  
int read_conf_file(gbdt_info_t& infbox, int argc, char* argv[])
{
	if(argv == NULL)
	{
		LOG_ERROR_("Parameter error.");
		return -1;
	}

	if(argc != 2 && argc != 15)
	{
		print_usage(stderr, argv[0]);
		return -1;
	}

	int ch;
	double random_feature_ratio = -1;
	char message[BUFFER_LENGTH];

	const char* short_options = "h:r:t:s:n:d:m:f:";

	const struct option long_options[]={
 	{"help", 0, NULL, 'h'},
 	{"sample_feature_ratio", 1, NULL, 'r'},
 	{"tree_num", 1, NULL, 't'},
 	{"shrink", 1, NULL, 's'},
 	{"min_node_size", 1, NULL, 'n'},
 	{"max_depth", 1, NULL, 'd'},
 	{"model_out", 1, NULL, 'm'},
 	{"train_file", 1, NULL, 'f'},
 	{NULL, 0, NULL, 0}};

 	init_info(infbox);

 	while((ch = getopt_long (argc, argv, short_options, long_options, NULL)) != -1)
 	{
		switch(ch)
 		{
			case 'h':
				if(argc == 2)
				{
					print_usage(stderr, argv[0]);
					return 1;
				}	
				else
					return -1;
			case 'r':
				if(sscanf(optarg, "%lf", &random_feature_ratio) != 1)
 				{
					LOG_ERROR_("Get random_feature_ratio config error.");
					return -1;
 				}
 				break;
			case 't':
				if(sscanf(optarg, "%d", &infbox.tree_num) != 1)
 				{
					LOG_ERROR_("Get tree_num config error.");
					return -1;
 				}
				break;
 			case 's':
				if(sscanf(optarg, "%lf", &infbox.shrink) != 1)
 				{
					LOG_ERROR_("Get shrink config error.");
					return -1;
 				}
 				break;
			case 'n':
 				if(sscanf(optarg, "%d", &infbox.gbdt_min_node_size) != 1)
 				{
					LOG_ERROR_("Get min_node_size config error.");
					return -1;
				}
				break;
 			case 'd':
 				if(sscanf(optarg, "%d", &infbox.gbdt_max_depth) != 1)
 				{
					LOG_ERROR_("Get max_depth config error.");
					return -1;
 				}
 				break;
 			case 'm':
 				if(strlen(optarg) <= BUFFER_LENGTH)
					strncpy(infbox.model_filename, optarg, BUFFER_LENGTH);
 				else
 				{
					LOG_ERROR_("Get model_filename config error.");
					return -1;
 				}
 				break;
 			case 'f':
 				if(strlen(optarg) <= BUFFER_LENGTH)
					strncpy(infbox.train_filename, optarg, BUFFER_LENGTH);
 				else
 				{
					LOG_ERROR_("Get train_filename config error.");
					return -1;
 				}
 				break;
 			case '?':
 				print_usage(stderr, argv[0]);
 				return -1;
 			case -1:
 				print_usage(stderr, argv[0]);
 				return -1;
 			default:
 				print_usage(stderr, argv[0]);
 				return -1;
 		}
 	}

 	ifstream in(infbox.train_filename);

 	if(in)
 	{
		string line;
		infbox.data_num = 0;
 		infbox.fea_num = 0;
 		int temp;

 		while(getline(in, line))
 		{
			temp = max_feature_label(line);
			if(infbox.fea_num < temp)
			{
				infbox.fea_num = temp;
			}
			infbox.data_num++;
 		}
 		in.close();
 	}
 	else
 	{
		LOG_ERROR_("Can't open the train data file.");
		return -1;
 	}

 	snprintf(message, BUFFER_LENGTH, "Data Num: %d", infbox.data_num);
 	LOG_NOTICE_(message);

 	infbox.fea_num++;
 	snprintf(message, BUFFER_LENGTH, "Feature Num: %d", infbox.fea_num);
 	LOG_NOTICE_(message);

 	infbox.sample_num = (int)(infbox.data_num * SAMPLE_RATIO);
 	infbox.rand_fea_num = (int)(infbox.fea_num * random_feature_ratio);

 	if(infbox.data_num <=0 || infbox.fea_num <= 0 || infbox.gbdt_max_depth <= 0 || infbox.gbdt_min_node_size <= 0 || infbox.rand_fea_num <= 0 || infbox.rand_fea_num > infbox.fea_num || infbox.sample_num <= 0 || infbox.shrink <=0 || infbox.shrink > 1 || infbox.tree_num <= 0)
 	{
        
		return -1;
 	}

 	return 0;
}

int fill_novalue_feature(double* x, int fea_num, int data_num, double* faverage)
{
	if(x == NULL || faverage == NULL)
 	{
		printf("Parameter error.");
		return -1;
 	}
 	double sum = 0, avg;
 	int cnt = 0;
 	int index;
 	int novalue_n;
 	vector<int> novalues;

 	for(int i = 0; i < fea_num; i++)
 	{
		sum = 0;
 		cnt = 0;
 		novalue_n = 0;

 		novalues.clear();
 		for(int j = 0; j < data_num; j++)
 		{
			index = j*fea_num + i;
			if(x[index] != NO_VALUE)
			{
				sum += x[index];
				cnt++;
			}
			else
			{
				novalues.push_back(index);
			}
		}

		if(cnt > 0)
		{
			avg = sum / (double)cnt;
			faverage[i] = avg;
			novalue_n = novalues.size();

			for(int j = 0; j < novalue_n; j++)
			{
				x[novalues[j]] = avg;
			}
		}
 	}

 	return 0;
}




//
//  gbdt_regression_train main routine;
//     parameter :
//			x_fea_value : 2-dim matrix ;  sample x feature 
//			y_result_score : 
//			infbox:  model training configuration parameters; 
//
gbdt_model_t* gbdt_regression_train(double *x_fea_value, double *y_result_score, gbdt_info_t infbox)
{
	if(x_fea_value == NULL || y_result_score == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return NULL;
 	}

 	bool failed = false;

 	gbdt_model_t* gbdt_model = (gbdt_model_t*)calloc(1, sizeof(gbdt_model_t));
 	if(gbdt_model == NULL)
 	{
		LOG_ERROR_("Failed to allocate memory.");
		return NULL;
 	}

 	gbdt_model->info = infbox;
 	gbdt_model->feature_average = (double*)calloc(gbdt_model->info.fea_num, sizeof(double));

 	if(gbdt_model->feature_average == NULL)
 	{
		LOG_ERROR_("Failed to allocate memory.");
		free_model(gbdt_model);
		return NULL;
 	}
 	gbdt_model->reg_forest = (gbdt_tree_t**) calloc(gbdt_model->info.tree_num, sizeof(gbdt_tree_t*));
 	if(gbdt_model->feature_average == NULL)//update wuyang
 	{
		LOG_ERROR_("Failed to allocate memory.");
		free_model(gbdt_model);
		return NULL;
 	}

	// compute the ¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿ gbdt_model->feature_average
 	fill_novalue_feature(x_fea_value, gbdt_model->info.fea_num, gbdt_model->info.data_num, gbdt_model->feature_average);

 	int nrnodes = 2* infbox.sample_num +1;

 	// gbdt_inf 

 	int* sample_in = (int *) calloc(infbox.data_num, sizeof(int));
 	//int* varUsed = (int *) calloc(infbox.fea_num, sizeof(int));
 	double* y_select  = (double *) calloc(infbox.sample_num, sizeof(double));
 	double* x_select = (double *) calloc(infbox.fea_num * infbox.sample_num, sizeof(double));
 	int* index = (int *) calloc(infbox.sample_num, sizeof(int));
 	bufset* data_set = (bufset*)calloc(1, sizeof(bufset));
 	gbdt_tree_t* pgbdtree = (gbdt_tree_t*) calloc (1, sizeof(gbdt_tree_t));

 	double* y_gradient  = (double *) calloc(infbox.data_num, sizeof(double));
 	double* y_pred = (double *) calloc(infbox.data_num, sizeof(double));
 	double* x_test = (double*) malloc (infbox.fea_num * sizeof(double));

 	if(index == NULL || sample_in == NULL || y_select == NULL || x_select == NULL || data_set == NULL || pgbdtree == NULL || y_gradient == NULL || y_pred == NULL || x_test == NULL)
 	{
		failed = true;
		LOG_ERROR_("Failed to allocate memory.");
		goto ROOT_EXIT;
 	}

 	for (int i=0; i< infbox.data_num; i++)
 	{
		y_gradient[i] = y_result_score[i];
		y_pred[i] = 0;
 	}

 	data_set->fea_pool     = (int *) calloc(infbox.fea_num, sizeof(int));
 	data_set->fvalue_list  = (double *) calloc(infbox.sample_num, sizeof(double));
 	data_set->y_list       = (double *) calloc(infbox.sample_num, sizeof(double));
 	data_set->fv           = (double *) calloc(infbox.sample_num, sizeof(double));
 	data_set->order_i      = (int *) calloc(infbox.sample_num, sizeof(int));

 	pgbdtree->nodestatus   = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->depth        = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->ndstart      = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->ndcount      = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->lson         = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->rson         = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->splitid      = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->splitvalue   = (double*) calloc (nrnodes, sizeof(double));
 	pgbdtree->ndavg        = (double*) calloc (nrnodes, sizeof(double));
 	pgbdtree->nodesize     = 0;

 	if(pgbdtree->nodestatus == NULL || pgbdtree->depth == NULL || pgbdtree->ndstart == NULL || pgbdtree->ndcount == NULL || pgbdtree->lson == NULL || pgbdtree->rson == NULL || pgbdtree->splitid == NULL || pgbdtree->splitvalue == NULL || pgbdtree->ndavg == NULL || data_set->fea_pool == NULL || data_set->fvalue_list == NULL || data_set->y_list == NULL || data_set->fv == NULL || data_set->order_i == NULL)
 	{
		failed = true;
 		LOG_ERROR_("Failed to allocate memory.");
 		goto EXIT;
 	}

 	//splitinfo* spinf = (splitinfo*)malloc(sizeof(splitinfo));

 	srand((unsigned)time(NULL));

 	for (int j = 0; j < infbox.tree_num; ++j) // 
 	{
#		ifdef DEBUG
 		printf("Tree: %d\n", j);
#		endif
 		for (int i = 0; i< infbox.sample_num; i++) {
			index[i] = i;
 		}

		for (int i = 0; i< infbox.data_num; i++) {
			sample_in[i] = 0;
 		}

		pgbdtree->nodesize = 0; // clear pgbtree

		/* build a single regression tree */
 		if(gbdt_single_tree_estimation(x_fea_value, y_gradient, infbox, data_set, index, pgbdtree, nrnodes) != 0)
 		{
			LOG_ERROR_("Training model failed.");
			goto EXIT;
 		}

		// copy pgbtree to gbdt_tree[j]
 		int ndsize = pgbdtree->nodesize;
 		gbdt_model->reg_forest[j] = (gbdt_tree_t*)calloc(1, sizeof(gbdt_tree_t));
 		if(gbdt_model->reg_forest[j] == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			goto EXIT;
 		}
		gbdt_model->reg_forest[j]->nodestatus = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->ndstart = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->ndcount = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->lson = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->rson = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->splitid = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->splitvalue = (double*) malloc (ndsize * sizeof(double));
 		gbdt_model->reg_forest[j]->ndavg = (double*) malloc (ndsize * sizeof(double));
 		gbdt_model->reg_forest[j]->nodesize = ndsize;

 		if(gbdt_model->reg_forest[j]->nodestatus == NULL || gbdt_model->reg_forest[j]->ndstart == NULL || gbdt_model->reg_forest[j]->ndcount == NULL || gbdt_model->reg_forest[j]->lson == NULL || gbdt_model->reg_forest[j]->rson == NULL || gbdt_model->reg_forest[j]->splitid == NULL || gbdt_model->reg_forest[j]->splitvalue == NULL || gbdt_model->reg_forest[j]->ndavg == NULL)
		{
			LOG_ERROR_("Failed to allocate memory.");
			goto EXIT;
 		}

 		for (int i=0; i<ndsize; i++)
 		{
			gbdt_model->reg_forest[j]->nodestatus[i] = pgbdtree->nodestatus[i];
 			gbdt_model->reg_forest[j]->ndstart[i] = pgbdtree->ndstart[i];
 			gbdt_model->reg_forest[j]->ndcount[i] = pgbdtree->ndcount[i];
 			gbdt_model->reg_forest[j]->lson[i] = pgbdtree->lson[i];
 			gbdt_model->reg_forest[j]->rson[i] = pgbdtree->rson[i];
 			gbdt_model->reg_forest[j]->splitid[i] = pgbdtree->splitid[i];
 			gbdt_model->reg_forest[j]->splitvalue[i] = pgbdtree->splitvalue[i];
 			gbdt_model->reg_forest[j]->ndavg[i] = pgbdtree->ndavg[i];
		// gbdt_tree[j]->nodesize = pgbtree->nodesize;
		}

 		for (int i=0; i< infbox.data_num; i++)
 		{
			for (int k=0; k<infbox.fea_num; k++)
 			{
				x_test[k] = x_fea_value[i * infbox.fea_num + k];
 			}
 			gbdt_tree_predict(x_test, gbdt_model->reg_forest[j], y_pred[i], infbox.shrink);
 			y_gradient[i] = y_result_score[i] - y_pred[i];
		}
 	}
 	/* ===== end of tree iterations =====*/
EXIT:
 	if(data_set->fea_pool != NULL)
 	{
 	free(data_set->fea_pool);
 	}
 	if(data_set->fvalue_list != NULL)
 	{
 	free(data_set->fvalue_list);
 	}
 	if(data_set->y_list != NULL)
 	{
 	free(data_set->y_list);
 	}
 	if(data_set->fv != NULL)
 	{
 	free(data_set->fv);
 	}
 	if(data_set->order_i != NULL)
 	{
 	free(data_set->order_i);
 	}

 	if(pgbdtree->nodestatus != NULL)
 	{
 	free(pgbdtree->nodestatus);
 	}
 	if(pgbdtree->depth != NULL)
 	{
 	free(pgbdtree->depth);
 	}
 	if(pgbdtree->ndstart != NULL)
 	{
 	free(pgbdtree->ndstart);
 	}
 	if(pgbdtree->ndcount != NULL)
 	{
 	free(pgbdtree->ndcount);
 	}
 	if(pgbdtree->lson != NULL)
 	{
 	free(pgbdtree->lson);
 	}
 	if(pgbdtree->rson != NULL)
 	{
 	free(pgbdtree->rson);
 	}
 	if(pgbdtree->splitid != NULL)
 	{
 	free(pgbdtree->splitid);
 	}
 	if(pgbdtree->splitvalue != NULL)
 	{
 	free(pgbdtree->splitvalue);
 	}
 	if(pgbdtree->ndavg != NULL)
 	{
 	free(pgbdtree->ndavg);
 	}

ROOT_EXIT:
 	if(sample_in != NULL)
 	{
 	free(sample_in);
 	}
 	if(index != NULL)
 	{
 	free(index);
 	}
 	if(y_select != NULL)
 	{
 	free(y_select);
 	}
 	if(x_select != NULL)
 	{
 	free(x_select);
 	}
 	if(data_set != NULL)
 	{
 	free(data_set);
 	}
 	if(pgbdtree != NULL)
 	{
 	free(pgbdtree);
 	}

 	if(y_gradient != NULL)
 	{
 	free(y_gradient);
 	}
 	if(y_pred != NULL)
 	{
 	free(y_pred);
 	}
 	if(x_test != NULL)
 	{
 	free(x_test);
 	}

 	if(!failed)
 	{
 	return gbdt_model;
 	}
 	else
 	{
 	free_model(gbdt_model);
 	return NULL;
 	}
}
int save_gbdt_info(gbdt_info_t infbox, FILE* model_fp)
{
	if(model_fp == NULL)
 	{
		printf("Parameter error.");
 		return -1;
 	}
 	if(fwrite(&infbox.tree_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}
 	if(fwrite(&infbox.fea_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}
 	if(fwrite(&infbox.data_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}

 	if(fwrite(&infbox.rand_fea_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}
 	if(fwrite(&infbox.shrink, sizeof(double), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}
 	
 	if(fwrite(&infbox.gbdt_min_node_size, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}
 	
 	if(fwrite(&infbox.gbdt_max_depth, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Save model file error!");
 		return -1;
 	}
 	
 	return 0;
}
int gbdt_save_reg_forest(FILE* model_fp, gbdt_tree_t** reg_forest, gbdt_info_t infbox)
{
	if(model_fp == NULL || reg_forest == NULL)
 	{
		LOG_ERROR_("Parameter error.");
 		return -1;
 	}

 	int nodesize;

 	for(int i = 0; i < infbox.tree_num; i++)
 	{
		nodesize = reg_forest[i]->nodesize;
 		if(fwrite(&nodesize, sizeof(int), 1, model_fp) != 1)
 		{
			LOG_ERROR_("Save model file error!");
 			return -1;
 		}
		if((int)fwrite(reg_forest[i]->nodestatus, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Save model file error!");
			return -1;
 		}
 		if((int)fwrite(reg_forest[i]->splitid, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Save model file error!");
 			return -1;
 		}
 		if((int)fwrite(reg_forest[i]->splitvalue, sizeof(double), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Save model file error!");
 			return -1;
 		}
 		if((int)fwrite(reg_forest[i]->ndavg, sizeof(double), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Save model file error!");
 			return -1;
 		}

		if((int)fwrite(reg_forest[i]->rson, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Save model file error!");
 			return -1;
 		}
 		if((int)fwrite(reg_forest[i]->lson, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Save model file error!");
 			return -1;
 		}
 	}
 	return 0;
}
int gbdt_save_model(gbdt_model_t* gbdt_model, char* model_filename)
{
	if(model_filename == NULL || gbdt_model == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}
 	FILE* model_fp = fopen(model_filename, "wb");

 	if(!model_fp)
 	{
		LOG_ERROR_("Can't open model file!");
		return -1;
 	}

 	if(save_gbdt_info(gbdt_model->info, model_fp) == -1)
 	{
		LOG_ERROR_("Save model file error!");
 		fclose(model_fp);
 		return -1;
 	}

 	if(gbdt_save_reg_forest(model_fp, gbdt_model->reg_forest, gbdt_model->info) == -1)
 	{
		LOG_ERROR_("Save model file error!");
 		fclose(model_fp);
 		return -1;
 	}

 	if(gbdt_model->feature_average != NULL)
 	{
		if((int)fwrite(gbdt_model->feature_average, sizeof(double), gbdt_model->info.fea_num, model_fp) != gbdt_model->info.fea_num)
 		{
			LOG_ERROR_("Save model file error!");
 			fclose(model_fp);
 			return -1;
 		}
 	}
 	else
 	{
		LOG_ERROR_("Save model file error!");
 		fclose(model_fp);
 		return -1;
 	}

	fclose(model_fp);

	return 0;
}
   
   
int gbdt_load_reg_forest(FILE* model_fp, gbdt_model_t* gbdt_model)
{
	if(model_fp == NULL || gbdt_model == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}

	gbdt_model->reg_forest = (gbdt_tree_t**) malloc(gbdt_model->info.tree_num * sizeof(gbdt_tree_t*));

	if(gbdt_model->reg_forest == NULL)
 	{
		LOG_ERROR_("Failed to allocate memory.");
		return -1;
 	}
 	gbdt_tree_t* prtree;
 	int nodesize, rsize;

	for(int i = 0; i < gbdt_model->info.tree_num; i++)
 	{
		rsize = fread(&nodesize, sizeof(int), 1, model_fp);
		if(rsize != 1)
		{
			LOG_ERROR_("Load model file error.");
			return -1;
		}	

		gbdt_model->reg_forest[i] = (gbdt_tree_t*) calloc (1, sizeof(gbdt_tree_t));
 		if(gbdt_model->reg_forest[i] == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}
		prtree = gbdt_model->reg_forest[i];
		prtree->nodestatus = (int*) malloc (nodesize * sizeof(int));
		if(prtree->nodestatus == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}
 		prtree->lson = (int*) malloc (nodesize * sizeof(int));
 		if(prtree->lson == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}
		prtree->rson = (int*) malloc (nodesize * sizeof(int));
 		if(prtree->rson == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}
 		prtree->splitid = (int*) malloc (nodesize * sizeof(int));
 		if(prtree->splitid == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}
 		prtree->splitvalue = (double*) malloc (nodesize * sizeof(double));
 		if(prtree->splitvalue == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}
 		prtree->ndavg = (double*) malloc (nodesize * sizeof(double));
 		if(prtree->ndavg == NULL)
 		{
			LOG_ERROR_("Failed to allocate memory.");
			return -1;
 		}

		prtree->nodesize = nodesize;

		if((int)fread(prtree->nodestatus, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Load model file error!");
			return -1;
 		}
 		if((int)fread(prtree->splitid, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Load model file error!");
			return -1;
 		}
 		if((int)fread(prtree->splitvalue, sizeof(double), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Load model file error!");
			return -1;
 		}
 		if((int)fread(prtree->ndavg, sizeof(double), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Load model file error!");
			return -1;
 		}
 		if((int)fread(prtree->rson, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Load model file error!");
			return -1;
 		}
 		if((int)fread(prtree->lson, sizeof(int), nodesize, model_fp) != nodesize)
 		{
			LOG_ERROR_("Load model file error!");
			return -1;
 		}
 		prtree->ndstart = NULL;
 		prtree->ndcount = NULL;
	}
	return 0;
}
int load_gbdt_info(gbdt_info_t* pinfbox, FILE* model_fp)
{
	if(pinfbox == NULL || model_fp == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}
 	if(fread(&pinfbox->tree_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}
 	if(fread(&pinfbox->fea_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}

	if(fread(&pinfbox->data_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}

 	if(fread(&pinfbox->rand_fea_num, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}

 	if(fread(&pinfbox->shrink, sizeof(double), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}

 	if(fread(&pinfbox->gbdt_min_node_size, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}

 	if(fread(&pinfbox->gbdt_max_depth, sizeof(int), 1, model_fp) != 1)
 	{
		LOG_ERROR_("Load model file error!");
		return -1;
 	}

 	return 0;
}
gbdt_model_t* gbdt_load_model(char* model_file)
{
	if(!model_file)
 	{
		LOG_ERROR_("Parameter error.");
		return NULL;
 	}
 	FILE* model_fp = fopen(model_file, "rb");

 	if(!model_fp)
 	{
		LOG_ERROR_("Can't open model file!");
		return NULL;
 	}

	gbdt_model_t* gbdt_model =  (gbdt_model_t*)calloc(1, sizeof(gbdt_model_t));
 	if(!gbdt_model)
 	{
		LOG_ERROR_("Failed to allocate memory.");
 		fclose(model_fp);
 		return NULL;
 	}
 	if(load_gbdt_info(&gbdt_model->info, model_fp) != 0)
 	{
		LOG_ERROR_("Load model file error!");
 		free_model(gbdt_model);
 		fclose(model_fp);
 		return NULL;
 	}
	if(gbdt_load_reg_forest(model_fp, gbdt_model) == -1)
 	{
		LOG_ERROR_("Load model file error!");
 		free_model(gbdt_model);
 		fclose(model_fp);
 		return NULL;
 	}

	gbdt_model->feature_average = (double*)malloc(gbdt_model->info.fea_num * sizeof(double));
 	if(gbdt_model->feature_average == NULL)
 	{
		LOG_ERROR_("Load model file error!");
 		free_model(gbdt_model);
 		fclose(model_fp);
 		return NULL;
 	}
	if((int)fread(gbdt_model->feature_average, sizeof(double), gbdt_model->info.fea_num, model_fp) != gbdt_model->info.fea_num)
 	{
		LOG_ERROR_("Load model file error!");
 		free(gbdt_model->feature_average);
 		free_model(gbdt_model);
 		fclose(model_fp);
 		return NULL;
 	}

 	fclose(model_fp);

	return gbdt_model;
}

int free_model(gbdt_model_t*& gbdt_model)
{
 if(gbdt_model == NULL)
 {
 return 1;
 }

 gbdt_tree_t* prtree;

 if(gbdt_model->reg_forest != NULL)
 {
 for(int i = 0; i < gbdt_model->info.tree_num; i++)
 {
 prtree = gbdt_model->reg_forest[i];
 if(prtree != NULL)
 {
 if(prtree->nodestatus != NULL)
 {
 free(prtree->nodestatus);
 prtree->nodestatus = NULL;
 }
 if(prtree->depth != NULL)
 {
 free(prtree->depth);
 prtree->depth = NULL;
 }
 if(prtree->lson != NULL)
 {
 free(prtree->lson);
 prtree->lson = NULL;
 }
 if(prtree->rson != NULL)
 {
 free(prtree->rson);
 prtree->rson = NULL;
 }
 if(prtree->splitid != NULL)
 {
 free(prtree->splitid);
 prtree->splitid = NULL;
 }
 if(prtree->splitvalue != NULL)
 {
 free(prtree->splitvalue);
 prtree->splitvalue = NULL;
 }
 if(prtree->ndavg != NULL)
 {
 free(prtree->ndavg);
 prtree->ndavg = NULL;
 }
 if(prtree->ndstart != NULL)
 {
 free(prtree->ndstart);
 prtree->ndstart = NULL;
 }
 if(prtree->ndcount != NULL)
 {
 free(prtree->ndcount);
 prtree->ndcount = NULL;
 }
 free(gbdt_model->reg_forest[i]);
 gbdt_model->reg_forest[i] = NULL;
 }
 }
 free(gbdt_model->reg_forest);
 gbdt_model->reg_forest = NULL;
 }

 if(gbdt_model->feature_average != NULL)
 {
 free(gbdt_model->feature_average);
 gbdt_model->feature_average = NULL;
 }

 free(gbdt_model);
 gbdt_model = NULL;

 return 1;
}
   
#define qsort_Index
#define NUMERIC double
void R_qsort_I(double *v, int *I, int i, int j)
{
	/* Orders v[] increasingly. Puts into I[] the permutation vector:
 	*  new v[k] = old v[I[k]]
 	* Only elements [i : j]  (in 1-indexing !)  are considered.
 	*/

 	int il[31], iu[31];
 	NUMERIC vt, vtt;
 	double R = 0.375;
 	int ii, ij, k, l, m;
#ifdef qsort_Index
	int it, tt;
#endif


	/* 1-indexing for I[], v[]  (and `i' and `j') : */
	--v;
#ifdef qsort_Index
	--I;
#endif

	ii = i;/* save */
 	m = 1;

L10:
 if (i < j) {
 if (R < 0.5898437) R += 0.0390625; else R -= 0.21875;
L20:
 k = i;
 /* ij = (j + i) >> 1; midpoint */
 ij = i + (int)((j - i)*R);
#ifdef qsort_Index
 it = I[ij];
#endif
 vt = v[ij];
 if (v[i] > vt) {
#ifdef qsort_Index
 I[ij] = I[i]; I[i] = it; it = I[ij];
#endif
 v[ij] = v[i]; v[i] = vt; vt = v[ij];
 }
 /* L30:*/
 l = j;
 if (v[j] < vt) {
#ifdef qsort_Index
 I[ij] = I[j]; I[j] = it; it = I[ij];
#endif
 v[ij] = v[j]; v[j] = vt; vt = v[ij];
 if (v[i] > vt) {
#ifdef qsort_Index
 I[ij] = I[i]; I[i] = it; it = I[ij];
#endif
 v[ij] = v[i]; v[i] = vt; vt = v[ij];
 }
 }

 for(;;) { /*L50:*/
 //do l--;  while (v[l] > vt);
 l--;for(;v[l]>vt;l--);


#ifdef qsort_Index
 tt = I[l];
#endif
 vtt = v[l];
 /*L60:*/ 
 //do k++;  while (v[k] < vt);
 k=k+1;for(;v[k]<vt;k++);

 if (k > l) break;

 /* else (k <= l) : */
#ifdef qsort_Index
 I[l] = I[k]; I[k] =  tt;
#endif
 v[l] = v[k]; v[k] = vtt;
 }

 m++;
 if (l - i <= j - k) {
 /*L70: */
 il[m] = k;
 iu[m] = j;
 j = l;
 }
 else {
 il[m] = i;
 iu[m] = l;
 i = k;
 }
 }else { /* i >= j : */

L80:
 if (m == 1) return;

 /* else */
 i = il[m];
 j = iu[m];
 m--;
 }

 if (j - i > 10)  goto L20;

 if (i == ii)  goto L10;

 --i;
L100:
 do {
 ++i;
 if (i == j) {
 goto L80;
 }
#ifdef qsort_Index
 it = I[i + 1];
#endif
 vt = v[i + 1];
 } while (v[i] <= vt);

 k = i;

 do { /*L110:*/
#ifdef qsort_Index
 I[k + 1] = I[k];
#endif
 v[k + 1] = v[k];
 --k;
 } while (vt < v[k]);

#ifdef qsort_Index
 I[k + 1] = it;
#endif
 v[k + 1] = vt;
 goto L100;
}
   
   
   
   
   
   
   
   
   
   
