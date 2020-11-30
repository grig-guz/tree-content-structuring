#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午8:37

import sys
from os.path import isfile
import nltk
from nltk import Tree
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from dataset.utils.document import Doc
from dataset.utils.span import SpanNode, DepSpanNode, HiraoDepNode
import dgl


class ConstituencyTree(object):
    
    def __init__(self, fdis=None, fmerge=None):
        self.fdis = fdis
        self.fmerge = fmerge
        self.binary = True
        self.tree, self.doc = None, None

    def build(self):
        """ Build BINARY Constituency tree
        """
        with open(self.fdis) as fin:
            text = fin.read()
        # Build Constituency as annotation
        self.tree = ConstituencyTree.build_tree(text)
        # Binarize it
        self.tree = ConstituencyTree.binarize_tree(self.tree)
        # Read doc file
        if isfile(self.fmerge):
            doc = Doc()
            doc.read_from_fmerge(self.fmerge)
            self.doc = doc
        else:
            raise IOError("File doesn't exist: {}".format(self.fmerge))
        ConstituencyTree.down_prop(self.tree)
        ConstituencyTree.back_prop(self.tree, self.doc)

    def decode_rst_tree(self):
        """ Decoding Shift-reduce actions and span relations from a binary Constituency tree
        """
        # Start decoding
        post_nodelist = ConstituencyTree.postorder_DFT(self.tree, [])
        action_list = []
        relation_list = []
        for node in post_nodelist:
            if (node.lnode is None) and (node.rnode is None):
                action_list.append(('Shift', None))
                relation_list.append(None)
            elif (node.lnode is not None) and (node.rnode is not None):
                form = node.form
                if (form == 'NN') or (form == 'NS'):
                    relation = ConstituencyTree.extract_relation(node.rnode.relation)
                else:
                    relation = ConstituencyTree.extract_relation(node.lnode.relation)
                action_list.append(('Reduce', form))
                relation_list.append(relation)
            else:
                raise ValueError("Can not decode Shift-Reduce action")
        return action_list, relation_list

    def convert_node_to_str(self, node, sep=' '):
        text = node.text
        words = [self.doc.token_dict[tidx].word for tidx in text]
        return sep.join(words)        
    
    @staticmethod
    def hirao_convert_to_dependency(node, doc):
        # Node depth for each node
        node.nuclearity = 'Root'
        constit_nodes = ConstituencyTree.postorder_DFT(node, [])
        ConstituencyTree.determine_levels(node)
        # Left-to-right sequence of leaf (EDU) nodes
        const_leaf_list = ConstituencyTree.get_leaf_nodes(node)
        for i, node in enumerate(const_leaf_list):
            node.idx = i
        # Ok
        dep_leaf_list = [HiraoDepNode(l.idx, l.level, 
                                      " ".join([doc.token_dict[token].word for token in l.text])) for l in const_leaf_list]
        tree_heads = []
        
        for dep_edu, const_edu in zip(dep_leaf_list, const_leaf_list):
            closest_S_ancestor, is_tree_head = ConstituencyTree.find_S_ancestor(const_edu)
            if is_tree_head:
                tree_heads.append(dep_edu)
            else:
                head_idx = ConstituencyTree.find_head(closest_S_ancestor)
                for dep_node in dep_leaf_list:
                    if dep_node.idx == head_idx:
                        dep_node.add_child(dep_edu)
                                
        
        tree_heads = sorted(tree_heads, key=lambda x: x.level, reverse=False)
        
        for head_idx in range(len(tree_heads)-1):
            tree_heads[0].add_child(tree_heads[head_idx+1])
            
        ConstituencyTree.determine_sentence_number(dep_leaf_list)
        ConstituencyTree.ensure_single_rooted_sentences(dep_leaf_list, const_leaf_list)
        return tree_heads[0], dep_leaf_list
            
    @staticmethod
    def determine_levels(node, n=0):
        node.level = n
        if node.lnode is None:
            return 
        else:
            if node.form == 'SN':
                node.lnode.nuclearity = 'Satellite'
                node.rnode.nuclearity = 'Nucleus'
            elif node.form == 'NS':
                node.lnode.nuclearity = 'Nucleus'
                node.rnode.nuclearity = 'Satellite'
            else:
                node.lnode.nuclearity = 'Nucleus'
                node.rnode.nuclearity = 'Nucleus'
            ConstituencyTree.determine_levels(node.lnode, (n+1))
            ConstituencyTree.determine_levels(node.rnode, (n+1))

    @staticmethod
    def get_leaf_nodes(node):
        if node.lnode is None:
            return [node]
        else:
            leaves = []
            leaves.extend(ConstituencyTree.get_leaf_nodes(node.lnode))
            leaves.extend(ConstituencyTree.get_leaf_nodes(node.rnode))
            return leaves
         
    @staticmethod
    def find_S_ancestor(const_node):
        # If its a root?
        if const_node.pnode is None:
            return const_node, True
        # If node is a nucleus
        if const_node.nuclearity == 'Nucleus' and const_node.pnode is not None:
            closest_S_ancestor, is_tree_head = ConstituencyTree.find_S_ancestor(const_node.pnode)
        else:
            closest_S_ancestor = const_node.pnode
            is_tree_head = False
        return closest_S_ancestor, is_tree_head
         
    @staticmethod
    def find_head(node):
        if node.lnode is None:
            return node.idx

        for child_node in [node.lnode, node.rnode]:
            if child_node.nuclearity == 'Nucleus':
                head_idx = ConstituencyTree.find_head(child_node)
                break
        return head_idx
    
    @staticmethod
    def determine_sentence_number(dep_node_list):
        complete_doc = '(-!EDU_BREAK!-) ' + ' (-!EDU_BREAK!-) '.join([node.text for node in dep_node_list])
        edu_index = 0
        for sent_idx, sentence in enumerate(nltk.sent_tokenize(complete_doc)):
            # -1 as the first 'EDU' is always empty
            for _ in range(len(sentence.split('(-!EDU_BREAK!-) '))-1):
                dep_node_list[edu_index].sentence = sent_idx
                edu_index += 1

    @staticmethod
    def ensure_single_rooted_sentences(dep_node_list, const_node_list):
        curr_sentence = 0
        curr_dep_nodes = []
        curr_const_nodes = []
        for idx, node in enumerate(dep_node_list):
            if node.sentence == curr_sentence:
                # If this is not a root and the head is not in the current sentence:
                if not node.pnode is None and node.pnode.sentence != curr_sentence:
                    curr_dep_nodes.append(node)
                    curr_const_nodes.append(const_node_list[idx])
            if idx < len(dep_node_list)-1:
                # If current node is the last one in the current sentence
                if dep_node_list[idx+1].sentence != curr_sentence:
                    ConstituencyTree.rearrange_children(curr_dep_nodes, curr_const_nodes)
                    curr_sentence += 1
                    curr_dep_nodes = []
                    curr_const_nodes = []
        ConstituencyTree.rearrange_children(curr_dep_nodes, curr_const_nodes)

    @staticmethod
    def rearrange_children(curr_dep_nodes, curr_const_nodes):
        if len(curr_dep_nodes) > 1:
            for idx, element in enumerate(curr_dep_nodes):
                element.dist_to_root = ConstituencyTree.get_distance_to_root(curr_dep_nodes[idx], n=0)
            curr_dep_nodes = sorted(curr_dep_nodes, key=lambda x: x.dist_to_root, reverse=False)
            for element in curr_dep_nodes[1:]:
                if not element.pnode is None:
                    element.pnode.remove_child(element)
                curr_dep_nodes[0].add_child(element)
                
    @staticmethod
    def get_distance_to_root(node, n=0):
        if node.pnode == None:
            return n
        else:
            n = ConstituencyTree.get_distance_to_root(node.pnode, (n+1))
            return n    
    
    @staticmethod
    def postorder_DFT_dgl_dep_hirao(tree, l_ch_graph, r_ch_graph, node_id=0):
        root, dep_nodes = tree
        num_nodes = len(dep_nodes)
        l_ch_graph.add_nodes(num_nodes)
        r_ch_graph.add_nodes(num_nodes)
        for node in dep_nodes:
            for lnode in node.lnodes:
                l_ch_graph.add_edge(node.idx, lnode.idx)
            for rnode in node.rnodes:
                r_ch_graph.add_edge(node.idx, rnode.idx)
        return root.idx, None
    
    @staticmethod
    def get_edu_node(tree):
        """ Get all left nodes. It can be used for generating training
            examples from gold Constituency tree

        :type tree: SpanNode instance
        :param tree: an binary Constituency tree
        """
        # Post-order depth-first traversal
        post_nodelist = ConstituencyTree.postorder_DFT(tree, [])
        # EDU list
        edulist = []
        for node in post_nodelist:
            if (node.lnode is None) and (node.rnode is None):
                edulist.append(node)
        return edulist

    @staticmethod
    def build_tree(text):
        """ Build tree from *.dis file

        :type text: string
        :param text: Constituency tree read from a *.dis file
        """
        tokens = text.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
        # print('tokens = {}'.format(tokens))
        queue = ConstituencyTree.process_text(tokens)
        #print('queue = {}'.format(queue))
        stack = []
        while queue:
            token = queue.pop(0)
            if token == ')':
                # If ')', start processing
                content = []  # Content in the stack
                while stack:
                    cont = stack.pop()
                    if cont == '(':
                        break
                    else:
                        content.append(cont)
                content.reverse()  # Reverse to the original (stack) order
                # Parse according to the first content word
                if len(content) < 2:
                    raise ValueError("content = {}".format(content))
                label = content.pop(0)
                if label in ['Root', 'Nucleus', 'Satellite']:
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'span':
                    # Merge
                    beginindex = int(content.pop(0))
                    endindex = int(content.pop(0))
                    stack.append(('span', beginindex, endindex))
                elif label == 'leaf':
                    # Merge
                    eduindex = int(content.pop(0))
                    ConstituencyTree.check_content(label, content)
                    stack.append(('leaf', eduindex, eduindex))
                elif label == 'rel2par':
                    # Merge
                    relation = content.pop(0)
                    ConstituencyTree.check_content(label, content)
                    stack.append(('relation', relation))
                elif label == 'text':
                    # Merge
                    txt = ConstituencyTree.create_text(content)
                    stack.append(('text', txt))
                else:
                    raise ValueError(
                        "Unrecognized parsing label: {} \n\twith content = {}\n\tstack={}\n\tqueue={}".format(label,
                                                                                                              content,
                                                                                                              stack,
                                                                                                              queue))
            else:
                # else, keep push into the stack
                stack.append(token)
        return stack[-1]

    @staticmethod
    def process_text(tokens):
        """ Preprocessing token list for filtering '(' and ')' in text,
            replaces them with -LB- and -RB- respectively
        :type tokens: list
        :param tokens: list of tokens
        """
        identifier = '_!'
        within_text = False
        for (idx, tok) in enumerate(tokens):
            if identifier in tok:
                for _ in range(tok.count(identifier)):
                    within_text = not within_text
            if ('(' in tok) and within_text:
                tok = tok.replace('(', '-LB-')
            if (')' in tok) and within_text:
                tok = tok.replace(')', '-RB-')
            tokens[idx] = tok
        return tokens

    @staticmethod
    def create_text(lst):
        """ Create text from a list of tokens

        :type lst: list
        :param lst: list of tokens
        """
        newlst = []
        for item in lst:
            item = item.replace("_!", "")
            newlst.append(item)
        text = ' '.join(newlst)
        # Lower-casing
        return text.lower()

    @staticmethod
    def check_content(label, c):
        """ Check whether the content is legal

        :type label: string
        :param label: parsing label, such 'span', 'leaf'

        :type c: list
        :param c: list of tokens
        """
        if len(c) > 0:
            raise ValueError("{} with content={}".format(label, c))

    @staticmethod
    def binarize_tree(tree):
        """ Convert a general Constituency tree to a binary Constituency tree

        :type tree: instance of SpanNode
        :param tree: a general Constituency tree
        """
        queue = [tree]
        while queue:
            node = queue.pop(0)
            queue += node.nodelist
            # Construct binary tree
            if len(node.nodelist) == 2:
                node.lnode = node.nodelist[0]
                node.rnode = node.nodelist[1]
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node
            elif len(node.nodelist) > 2:
                # Remove one node from the nodelist
                node.lnode = node.nodelist.pop(0)
                newnode = SpanNode(node.nodelist[0].prop)
                newnode.nodelist += node.nodelist
                # Right-branching
                node.rnode = newnode
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node
                # Add to the head of the queue
                # So the code will keep branching
                # until the nodelist size is 2
                queue.insert(0, newnode)
            # Clear nodelist for the current node
            node.nodelist = []
        return tree

    @staticmethod
    def back_prop(tree, doc):
        """ Starting from leaf node, propagating node
            information back to root node

        :type tree: SpanNode instance
        :param tree: an binary Constituency tree
        """
        tree_nodes = ConstituencyTree.BFTbin(tree)
        tree_nodes.reverse()
        for node in tree_nodes:
            if (node.lnode is not None) and (node.rnode is not None):
                # Non-leaf node
                node.edu_span = ConstituencyTree.__getspaninfo(node.lnode, node.rnode)
                node.text = ConstituencyTree.__gettextinfo(doc.edu_dict, node.edu_span)
                if node.relation is None:
                    # If it is a new node created by binarization
                    if node.prop == 'Root':
                        pass
                    else:
                        node.relation = ConstituencyTree.__getrelationinfo(node.lnode, node.rnode)
                node.form, node.nuc_span, node.nuc_edu = ConstituencyTree.__getforminfo(node.lnode, node.rnode)
                node.height = max(node.lnode.height, node.rnode.height) + 1
                node.max_depth = max(node.lnode.max_depth, node.rnode.max_depth)
                if node.form == 'NS':
                    node.child_relation = node.rnode.relation
                else:
                    node.child_relation = node.lnode.relation
                if doc.token_dict[node.lnode.text[0]].sidx == doc.token_dict[node.rnode.text[-1]].sidx:
                    node.level = 0
                elif doc.token_dict[node.lnode.text[0]].pidx == doc.token_dict[node.rnode.text[-1]].pidx:
                    node.level = 1
                else:
                    node.level = 2
            elif (node.lnode is None) and (node.rnode is not None):
                raise ValueError("Unexpected left node")
            elif (node.lnode is not None) and (node.rnode is None):
                raise ValueError("Unexpected right node")
            else:
                # Leaf node
                node.text = ConstituencyTree.__gettextinfo(doc.edu_dict, node.edu_span)
                node.height = 0
                node.max_depth = node.depth
                node.level = 0

    @staticmethod
    def down_prop(tree):
        """
        Starting from root node, propagating node information down to leaf nodes
        :param tree: SpanNode instance
        :param doc: Doc instance
        :return: root node
        """
        tree_nodes = ConstituencyTree.BFTbin(tree)
        root_node = tree_nodes.pop(0)
        root_node.depth = 0
        for node in tree_nodes:
            assert node.pnode.depth >= 0
            node.depth = node.pnode.depth + 1

    @staticmethod
    def BFT(tree):
        """ Breadth-first treavsal on general Constituency tree

        :type tree: SpanNode instance
        :param tree: an general Constituency tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            queue += node.nodelist
        return bft_nodelist

    @staticmethod
    def BFTbin(tree):
        """ Breadth-first treavsal on binary Constituency tree

        :type tree: SpanNode instance
        :param tree: an binary Constituency tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            if node.lnode is not None:
                queue.append(node.lnode)
            if node.rnode is not None:
                queue.append(node.rnode)
        return bft_nodelist

    @staticmethod
    def postorder_DFT(tree, nodelist):
        """ Post order traversal on binary Constituency tree

        :type tree: SpanNode instance
        :param tree: an binary Constituency tree

        :type nodelist: list
        :param nodelist: list of node in post order
        """
        if tree.lnode is not None:
            ConstituencyTree.postorder_DFT(tree.lnode, nodelist)
            ConstituencyTree.postorder_DFT(tree.rnode, nodelist)
        nodelist.append(tree)
        return nodelist

    @staticmethod
    def postorder_DFT_dgl(tree, graph, node_id=0, parent_id=0):
        """ Post order traversal on binary Constituency tree

        :type tree: SpanNode instance
        :param tree: an binary Constituency tree

        :type nodelist: list
        :param nodelist: list of node in post order
        """
        graph.add_nodes(1)
        if tree.pnode is not None:
            graph.add_edge(node_id, parent_id)
        new_node_id = node_id
        if tree.lnode is not None:
            new_node_id = ConstituencyTree.postorder_DFT_dgl(tree.lnode, graph, node_id+1, parent_id=node_id)
            new_node_id = ConstituencyTree.postorder_DFT_dgl(tree.rnode, graph, new_node_id+1, parent_id=node_id)
        return new_node_id

        
    @staticmethod
    def __getspaninfo(lnode, rnode):
        """ Get span size for parent node

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        try:
            edu_span = (lnode.edu_span[0], rnode.edu_span[1])
            return edu_span
        except TypeError:
            print(lnode.prop, rnode.prop)
            print(lnode.nuc_span, rnode.nuc_span)
            sys.exit()

    @staticmethod
    def __getforminfo(lnode, rnode):
        """ Get Nucleus/Satellite form and Nucleus span

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            nuc_span = lnode.edu_span
            nuc_edu = lnode.nuc_edu
            form = 'NS'
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            nuc_span = rnode.edu_span
            nuc_edu = rnode.nuc_edu
            form = 'SN'
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
            nuc_edu = lnode.nuc_edu
            form = 'NN'
        else:
            raise ValueError("")
        return form, nuc_span, nuc_edu

    @staticmethod
    def __getrelationinfo(lnode, rnode):
        """ Get relation information

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            relation = lnode.relation
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            relation = lnode.relation
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            relation = rnode.relation
        else:
            print('lnode.prop = {}, lnode.edu_span = {}'.format(lnode.prop, lnode.edu_span))
            print('rnode.prop = {}, lnode.edu_span = {}'.format(rnode.prop, rnode.edu_span))
            raise ValueError("Error when find relation for new node")
        return relation

    @staticmethod
    def __gettextinfo(edu_dict, edu_span):
        """ Get text span for parent node

        :type edu_dict: dict of list
        :param edu_dict: EDU from this document

        :type edu_span: tuple with two elements
        :param edu_span: start/end of EDU IN this span
        """
        # text = lnode.text + " " + rnode.text
        text = []
        for idx in range(edu_span[0], edu_span[1] + 1, 1):
            text += edu_dict[idx]
        # Return: A list of token indices
        return text

    @staticmethod
    def extract_relation(s, level=0):
        """ Extract discourse relation on different level
        """
        return "span"

    def get_parse(self):
        """ Get parse tree

        :type tree: SpanNode instance
        :param tree: an binary Constituency tree
        """
        parse = []
        node_list = [self.tree]
        while node_list:
            node = node_list.pop()
            if node == ' ) ':
                parse.append(' ) ')
                continue
            if (node.lnode is None) and (node.rnode is None):
                # parse.append(" ( EDU " + str(node.nuc_edu))
                parse.append(" ( EDU " + '_!' + self.convert_node_to_str(node, sep='_') + '!_')
            else:
                parse.append(" ( " + node.form)
                # get the relation from its satellite node
                if node.form == 'NN':
                    parse += "-" + ConstituencyTree.extract_relation(node.rnode.relation)
                elif node.form == 'NS':
                    parse += "-" + ConstituencyTree.extract_relation(node.rnode.relation)
                elif node.form == 'SN':
                    parse += "-" + ConstituencyTree.extract_relation(node.lnode.relation)
                else:
                    raise ValueError("Unrecognized N-S form")
            node_list.append(' ) ')
            if node.rnode is not None:
                node_list.append(node.rnode)
            if node.lnode is not None:
                node_list.append(node.lnode)
        return ''.join(parse)

def test_tree(tree_g):
    path="dataset path"
    tree = ConstituencyTree(fdis=path + 'auxDP_65543.out.dis', fmerge=path + 'auxDP_65543.out.merge')
    tree.build()
    print(tree.tree)
    print(ConstituencyTree.postorder_DFT_dgl(tree.tree,tree_g) == 12)
    return tree_g
